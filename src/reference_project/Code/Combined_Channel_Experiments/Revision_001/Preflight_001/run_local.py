"""CLI; original experiment workers are never invoked."""
import argparse
import json
import local_engine as e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-test", "prepare", "run", "verify", "report", "all", "status"))
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(e.status(), indent=2, ensure_ascii=False), flush=True)
        return
    with e.lock(args.command):
        if args.command == "self-test":
            from verify_local import numerical
            print(json.dumps(numerical(save=True), indent=2), flush=True)
        elif args.command == "prepare":
            m = e.prepare()
            print(json.dumps({"status": "FROZEN", "fingerprint": m["fingerprint"]}), flush=True)
        elif args.command in ("run", "all"):
            if args.command == "all":
                e.prepare()
            e.run()
            from verify_local import verify, audit_report
            print(json.dumps(verify(), indent=2), flush=True)
            e.summarize()
            print(json.dumps(audit_report(), indent=2), flush=True)
        elif args.command == "verify":
            from verify_local import verify
            print(json.dumps(verify(), indent=2), flush=True)
        else:
            from verify_local import audit_report
            e.summarize()
            print(json.dumps(audit_report(), indent=2), flush=True)


if __name__ == "__main__":
    main()
