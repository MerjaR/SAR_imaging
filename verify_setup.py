"""
Quick setup verification — run this to check everything works.
Usage: python verify_setup.py
"""

import sys

def main():
    print("Checking setup...\n")

    # Check Python version
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ required. You have {version.major}.{version.minor}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")

    # Check anthropic package
    try:
        import anthropic
        print(f"✅ anthropic package installed (v{anthropic.__version__})")
    except ImportError:
        print("❌ anthropic package not found. Run: pip install -r requirements.txt")
        sys.exit(1)

    # Check skyfield
    try:
        import skyfield
        print(f"✅ skyfield installed (v{skyfield.__version__})")
    except ImportError:
        print("❌ skyfield not found. Run: pip install -r requirements.txt")
        sys.exit(1)

    # Check requests
    try:
        import requests
        print(f"✅ requests installed (v{requests.__version__})")
    except ImportError:
        print("❌ requests not found. Run: pip install -r requirements.txt")
        sys.exit(1)

    # Load .env file if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ .env file loaded")
    except ImportError:
        print("⚠️  python-dotenv not found — skipping .env load")

    # Check API key
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set. Add it to your .env file.")
        sys.exit(1)
    print(f"✅ API key found (starts with {api_key[:8]}...)")

    # Test API connection
    try:
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say 'hello' in one word."}]
        )
        print(f"✅ API connection works! Claude says: {message.content[0].text}")
    except anthropic.AuthenticationError:
        print("❌ Invalid API key. Check your key at console.anthropic.com")
        sys.exit(1)
    except Exception as e:
        print(f"❌ API error: {e}")
        sys.exit(1)

    print("\n✅ All good! Your ICEYE agent is ready to run. 🛰️")


if __name__ == "__main__":
    main()