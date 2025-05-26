import os

def check_gemini_key():
    key = os.getenv("GEMINI_API_KEY")
    if not key or key == "AI_PLACEHOLDER_FOR_VERTEX_AI":
        print("\u274C GEMINI_API_KEY is missing or placeholder. Set it with:\n  export GEMINI_API_KEY=your-key-here")
        exit(1)
    print("\u2705 GEMINI_API_KEY is set.")

if __name__ == "__main__":
    check_gemini_key() 