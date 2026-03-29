from engine.live_runner import run_live_recommendation
import json

def main():
    action = run_live_recommendation()
    print(json.dumps(action, indent=2, default=str))

if __name__ == "__main__":
    main()
