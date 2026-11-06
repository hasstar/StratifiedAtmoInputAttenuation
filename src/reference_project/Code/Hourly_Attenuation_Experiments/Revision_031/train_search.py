"""Formal full-156 training; shared fixed-epoch fit for subsequent layer/RFE jobs."""
from __future__ import annotations

import argparse
import copy
import json
import math
import time

import core as c


def weights_digest(state):
    digest = c.hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_model(result):
    path = c.OUT / result['model_file']
    c.require(c.sha(path) == result['model_sha256'], 'Committed model SHA mismatch')
    bundle = c.load_tensor(path)
    c.require(bundle['identity'] == result['identity'], 'Checkpoint identity mismatch')
    c.set_seed()
    net = c.AtmosphericTabM(bundle['names'], bundle['config'])
    net.load_state_dict(bundle['weights'], strict=True)
    return net, bundle


def fit_job(job_key, cfg, names, train, manifest, hold=None, fixed_epochs=None):
    search = fixed_epochs is None
    c.require(not search or hold is not None, 'Search requires official validation')
    directory = c.OUT / 'jobs' / job_key
    directory.mkdir(parents=True, exist_ok=True)
    identity = c.fingerprint(dict(job_key=job_key, config=cfg, names=names,
        source=manifest['fingerprint'], train_hours=train.hour_bjt.astype(str).tolist(),
        hold_hours=None if hold is None else hold.hour_bjt.astype(str).tolist(),
        seed=c.SEED, fixed_epochs=fixed_epochs))
    if (directory / 'result.json').exists():
        old = c.read(directory / 'result.json')
        c.require(old['identity'] == identity and c.sha(c.OUT / old['model_file']) == old['model_sha256'], 'Changed job on resume')
        return old
    started = time.perf_counter()
    x, prep = c.fit_preprocess(train, names)
    xh = None if hold is None else c.transform(hold, names, prep)
    c.set_seed()
    net = c.AtmosphericTabM(names, cfg)
    optimizer = c.torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    generator = c.torch.Generator().manual_seed(c.SEED)
    xx = c.torch.from_numpy(x)
    yy = c.torch.as_tensor((train[c.TARGET].to_numpy(float) - prep['y_mean']) / prep['y_scale'], dtype=c.torch.float32)
    limit = manifest['max_epochs'] if search else fixed_epochs
    history, first_epoch, best_epoch, stale, best_score, best_weights, prior_seconds = [], 1, 0, 0, float('inf'), None, 0.
    recovery = directory / 'recovery.pt'
    if recovery.exists():
        saved = c.load_tensor(recovery)
        c.require(saved['identity'] == identity and saved['preprocess'] == prep, 'Invalid recovery state')
        net.load_state_dict(saved['weights'])
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator_rng'])
        c.torch.set_rng_state(saved['torch_rng'])
        history, first_epoch = saved['history'], saved['epoch'] + 1
        best_epoch, best_score, stale = saved['best_epoch'], saved['best_score'], saved['stale']
        best_weights, prior_seconds = saved['best_weights'], saved['fit_seconds']
    c.progress('formal_search' if search else 'fixed_refit', job=job_key, n_features=len(names),
               starting_epoch=first_epoch, epoch_limit=limit, config_id=cfg['id'], timing_pilot=False)
    for epoch in range(first_epoch, limit + 1):
        if search and stale >= manifest['patience']:
            break
        net.train()
        order = c.torch.randperm(len(xx), generator=generator)
        loss_sum = 0.
        for index in order.split(manifest['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            members = net(xx[index])
            loss = (members - yy[index, None]).square().mean()
            c.require(c.torch.isfinite(loss).item(), 'Nonfinite training loss')
            loss.backward()
            c.require(all(p.grad is None or c.torch.isfinite(p.grad).all().item() for p in net.parameters()), 'Nonfinite gradient')
            optimizer.step()
            loss_sum += float(loss.item()) * len(index)
        train_pred = c.predict(net, x, prep)
        train_metrics = c.metrics(train, train_pred)
        hold_metrics = None if hold is None else c.metrics(hold, c.predict(net, xh, prep))
        score = c.objective(hold_metrics) if search else None
        if not search or score < best_score:
            best_weights = copy.deepcopy(net.state_dict())
            best_epoch, best_score, stale = epoch, score if search else 0., 0
        else:
            stale += 1
        elapsed = prior_seconds + time.perf_counter() - started
        row = dict(epoch=epoch, member_mse_standardized=loss_sum / len(xx), train_metrics=train_metrics,
                   holdout_metrics=hold_metrics, official_validation_objective_db=score,
                   fit_seconds=elapsed, selected_epoch_so_far=best_epoch)
        history.append(row)
        c.append(directory / 'epoch_events.jsonl', dict(row, utc=c.now(), identity=identity))
        if search or epoch % 5 == 0 or epoch == limit:
            c.save_tensor(recovery, dict(identity=identity, preprocess=prep, epoch=epoch,
                weights=net.state_dict(), optimizer=optimizer.state_dict(), generator_rng=generator.get_state(),
                torch_rng=c.torch.get_rng_state(), history=history, best_epoch=best_epoch,
                best_score=best_score, best_weights=best_weights, stale=stale, fit_seconds=elapsed), intermediate=True)
        c.progress('formal_search' if search else 'fixed_refit', job=job_key, epoch=epoch,
                   n_features=len(names), config_id=cfg['id'], train_rmse_db=train_metrics['overall']['rmse_db'],
                   validation_objective_db=score, elapsed_fit_seconds=elapsed)
    c.require(best_weights is not None and best_epoch > 0, 'No trained checkpoint')
    net.load_state_dict(best_weights)
    bundle = dict(identity=identity, source_fingerprint=manifest['fingerprint'], names=names, config=cfg,
                  seed=c.SEED, selected_epoch=best_epoch, preprocess=prep, weights=best_weights)
    model_path = directory / 'model.pt'
    if model_path.exists():
        saved = c.load_tensor(model_path)
        c.require(saved['identity'] == identity and weights_digest(saved['weights']) == weights_digest(best_weights),
                  'Orphaned final checkpoint differs; preserve and diagnose')
    else:
        c.save_tensor(model_path, bundle)
    replay = c.load_tensor(model_path)
    net.load_state_dict(replay['weights'])
    result = dict(identity=identity, source_fingerprint=manifest['fingerprint'], job_key=job_key,
                  config=cfg, feature_names=names, n_features=len(names), seed=c.SEED,
                  trained_epochs=len(history), selected_epoch=best_epoch, search=search,
                  stop_reason=('validation_patience' if stale >= manifest['patience'] else 'epoch_cap') if search else 'fixed_frozen_epochs',
                  best_validation_objective_db=best_score if search else None, round_history=history,
                  fit_seconds=prior_seconds + time.perf_counter() - started,
                  model_file=str(model_path.relative_to(c.OUT)), model_sha256=c.sha(model_path),
                  network_weights_sha256=weights_digest(best_weights), preprocess=prep,
                  parameter_count=sum(p.numel() for p in net.parameters()),
                  training=c.prediction_record(train, c.predict(net, x, prep)),
                  holdout=None if hold is None else c.prediction_record(hold, c.predict(net, xh, prep)),
                  completed_utc=c.now(), test_used_for_fit_or_selection=False)
    c.emit(directory / 'result.json', result)
    c.append(c.OUT / 'fit_receipts.jsonl', dict(job_key=job_key, identity=identity, model_sha256=result['model_sha256'], utc=c.now()))
    return result


def run(manifest):
    c.require(not (c.OUT / 'search_test_opened.json').exists() or (c.OUT / 'search_selection.json').exists(), 'Test opened before selection')
    train, val = c.frame('train', manifest), c.frame('validation', manifest)
    c.require(set(train.hour_bjt).isdisjoint(val.hour_bjt), 'Train/validation exact hours overlap')
    records = [fit_job(f'search_{cfg["id"]:02d}', cfg, manifest['feature_names'], train, manifest, val) for cfg in manifest['configs']]
    chosen = min(records, key=lambda row:(row['best_validation_objective_db'], row['config']['id']))
    selection = dict(source_fingerprint=manifest['fingerprint'], config=chosen['config'],
                     fixed_epochs=chosen['selected_epoch'], seed=c.SEED,
                     selected_job=chosen['job_key'], selected_model_file=chosen['model_file'],
                     selected_model_sha256=chosen['model_sha256'],
                     candidates=[dict(id=r['config']['id'], objective_db=r['best_validation_objective_db'],
                                      selected_epoch=r['selected_epoch'], result_sha256=c.sha(c.OUT/'jobs'/r['job_key']/'result.json')) for r in records],
                     rule=manifest['selection_objective'], test_used=False, frozen_utc=c.now())
    target = c.OUT / 'search_selection.json'
    if target.exists():
        saved = c.read(target)
        c.require({k:v for k,v in saved.items() if k != 'frozen_utc'} == {k:v for k,v in selection.items() if k != 'frozen_utc'}, 'Selection changed')
    else:
        c.emit(target, selection)
    import verify_search
    verify_search.verify(manifest)
    c.progress('full156_selection_verified', selected_id=chosen['config']['id'], fixed_epochs=chosen['selected_epoch'], test_opened=False)
    score_full156(manifest)


def score_full156(manifest):
    c.source_manifest()
    selection = c.read(c.OUT / 'search_selection.json')
    c.require(c.read(c.OUT / 'search_qa.json')['status'] == 'PASS', 'Search QA gate')
    test = c.frame('test', manifest, allow_test=True)
    c.require([c.metrics(test, c.np.zeros(len(test)))[g]['n'] for g in c.WEATHER] == [687,322,51,158,156], 'Unexpected test weather counts')
    if not (c.OUT/'search_test_opened.json').exists():
        c.emit(c.OUT/'search_test_opened.json', dict(utc=c.now(), selection_sha256=c.sha(c.OUT/'search_selection.json'),
                scope='Only frozen full156 baseline and selected model; no retuning allowed'))
    for candidate in sorted({0, selection['config']['id']}):
        target = c.OUT / 'full156' / f'candidate_{candidate:02d}_test.json'
        if target.exists():
            continue
        record = c.read(c.OUT / 'jobs' / f'search_{candidate:02d}' / 'result.json')
        net, bundle = load_model(record)
        pred = c.predict(net, c.transform(test, bundle['names'], bundle['preprocess']), bundle['preprocess'])
        c.emit(target, dict(model_sha256=record['model_sha256'], config=record['config'],
                          selection_sha256=c.sha(c.OUT/'search_selection.json'),
                          result=c.prediction_record(test, pred), fit_calls=0, utc=c.now()))
    import verify_search
    verify_search.verify_full156(manifest)
    c.progress('full156_complete', layer_rfe_complete=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('start', 'resume'))
    args = parser.parse_args()
    with c.writer_lock('formal-search'):
        if args.command == 'start':
            from self_tests import run_tests
            run_tests()
        manifest = c.source_manifest(create=args.command == 'start')
        c.require(c.read(c.OUT/'self_test.json')['status'] == 'PASS', 'Self-test gate')
        run(manifest)


if __name__ == '__main__':
    main()
