import os
import sys

_env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_file):
    try:
        with open(_env_file, "r") as _ef:
            for _line in _ef:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
    except Exception:
        pass

def main():
    print("=" * 70)
    print("   Google Earth Engine Authentication Helper")
    print("   MOIL Limited — Manganese Prospectivity Exploration Portal")
    print("=" * 70)

    try:
        import ee
    except ImportError:
        print("\nERROR: earthengine-api is not installed in this Python environment.")
        print(f"Current Python: {sys.executable}")
        print("Run: pip install earthengine-api")
        sys.exit(1)

    print(f"\nPython Environment: {sys.executable}")
    print(f"Earth Engine API Version: {ee.__version__}")
    print("\nInitiating Earth Engine interactive authentication flow...")
    print("A browser window will open, or an authorization URL will be displayed.\n")

    try:
        ee.Authenticate()
        print("\nSUCCESS: Credentials saved to ~/.config/earthengine/credentials.")
    except Exception as e:
        print(f"\nAuthentication flow failed or was cancelled: {e}")
        sys.exit(1)

    print("\nRunning GEE Health Check...")
    project = (
        os.environ.get("EARTHENGINE_PROJECT")
        or os.environ.get("EARTH_ENGINE_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
    )

    try:
        if project:
            ee.Initialize(project=project)
            print(f"Initialized Earth Engine with project: {project}")
        else:
            ee.Initialize()
            print("Initialized Earth Engine with default credentials project.")

        test_val = ee.Number(1).getInfo()
        print(f"Health Check Query result: ee.Number(1).getInfo() = {test_val}")
        print("\n" + "=" * 70)
        print("   SUCCESS! Google Earth Engine is ready and verified.")
        print("   The Streamlit application will now automatically extract real")
        print("   Sentinel-1, Sentinel-2, and SRTM observations.")
        print("=" * 70)

    except Exception as e:
        err = str(e)
        print("\n" + "!" * 70)
        print(f"Credentials were saved, but GEE project initialization reported:")
        print(f"  {err}")
        if "Please provide a project" in err or "NO_PROJECT" in err:
            print("\nPlease set your Google Cloud project ID using:")
            print("  export EARTHENGINE_PROJECT='your-google-cloud-project-id'")
            print("or run:")
            print("  python3 -c \"import ee; ee.Initialize(project='your-project-id')\"")
        print("!" * 70)

if __name__ == "__main__":
    main()
