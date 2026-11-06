"""Fit and independently verify the CV-frozen marginal on all train hours."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from fit_marginals import CLASSES, fit, frozen_manifest, load_observations, serializable_model
from window_diagnostics import OUT, now, sha256, worker_lock, write_json

DESTINATION = OUT / "marginal_model"


def verify() -> None:
    saved = json.loads((DESTINATION / "model.json").read_text(encoding="utf-8"))
    if (saved["selection_sha256"] != sha256(OUT / "marginal_selection.json")
            or saved["marginal_manifest_sha256"] != sha256(OUT / "marginal_manifest.json")
            or saved["builder_sha256"] != sha256(Path(__file__))):
        raise RuntimeError("Frozen final marginal fingerprint mismatch")
    means = {}
    for label, parameters in saved["components"].items():
        family = parameters["family"]
        if family == "empirical":
            p = DESTINATION / parameters["support_file"]
            if sha256(p) != parameters["support_sha256"]:
                raise RuntimeError("Empirical support changed")
            with np.load(p) as content:
                q, cdf = content["q"], content["cdf"]
            weight = np.diff(np.r_[0., cdf])
            if (not np.isfinite(q).all() or (q <= 0).any() or (np.diff(q) < 0).any()
                    or (weight <= 0).any() or cdf[-1] != 1.):
                raise RuntimeError("Invalid empirical probability support")
            means[label] = float(np.sum(weight * q))
        elif family == "lognormal":
            means[label] = math.exp(parameters["mu"] + parameters["sigma"] ** 2 / 2)
        elif family == "gamma":
            means[label] = parameters["shape"] * parameters["scale"]
        else:
            means[label] = parameters["envelope_scale"] ** 2 * (parameters["b"] ** 2 + 2)
        if not math.isclose(means[label], 1., rel_tol=0, abs_tol=1e-10):
            raise RuntimeError("Saved final distribution does not preserve unit power")
    if sum(p["n_hours"] for p in saved["components"].values()) != 2404:
        raise RuntimeError("Final model train-hour count mismatch")
    write_json(OUT / "marginal_model_qa.json", {
        "status": "PASS", "completed_utc": now(), "candidate": saved["candidate"],
        "train_hours": 2404, "unit_power_moment_by_component": means,
        "max_power_moment_error": max(abs(value - 1.) for value in means.values()),
        "model_json_sha256": sha256(DESTINATION / "model.json"),
        "note": "Marginal model frozen; no temporal generator or formal test scoring yet.",
    })
    print(json.dumps(means, indent=2), flush=True)


def main() -> None:
    selection = json.loads((OUT / "marginal_selection.json").read_text(encoding="utf-8"))
    qa = json.loads((OUT / "marginal_cv_qa.json").read_text(encoding="utf-8"))
    if (selection["status"] != "frozen_marginal_family" or not selection["grid_rank_winner_agrees"]
            or qa["status"] != "PASS" or selection["marginal_cv_qa_sha256"] != sha256(OUT / "marginal_cv_qa.json")):
        raise RuntimeError("CV selection is not ready to freeze")
    if (DESTINATION / "model.json").exists():
        verify()
        print("Final marginal already saved and verified", flush=True)
        return
    manifest, assignment = frozen_manifest()
    family, condition = selection["candidate"].split("__")
    DESTINATION.mkdir(exist_ok=True)
    with worker_lock("freeze_marginal.py"):
        observations = load_observations(assignment, manifest)
        labels = ("global",) if condition == "global" else CLASSES
        components = {}
        for label in labels:
            group = observations if label == "global" else [x for x in observations if x.event == label]
            model = fit(family, group)
            parameters = serializable_model(model)
            if family == "empirical":
                path = DESTINATION / f"{label}_power_support.npz"
                np.savez_compressed(path, q=model["support_q"], cdf=model["support_cdf"])
                parameters.update(support_file=path.name, support_sha256=sha256(path))
            components[label] = parameters
            print(f"Saved final marginal/{label}: {len(group)} train hours", flush=True)
        write_json(DESTINATION / "model.json", {
            "candidate": selection["candidate"], "completed_utc": now(),
            "selection_sha256": sha256(OUT / "marginal_selection.json"),
            "marginal_manifest_sha256": sha256(OUT / "marginal_manifest.json"),
            "builder_sha256": sha256(Path(__file__)),
            "definition": "Q=P_1s/Pbar_h; F=-10log10(Q); each component has E(Q)=1",
            "components": components,
        })
    verify()


if __name__ == "__main__":
    main()
