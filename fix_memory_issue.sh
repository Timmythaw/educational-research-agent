#!/bin/bash
# Fix memory corruption issue with FAISS

echo "🔧 Fixing memory corruption issue..."

# Activate virtual environment
source .venv/bin/activate

# Reinstall FAISS and numpy with specific versions
echo "📦 Reinstalling FAISS and NumPy..."
pip uninstall -y faiss-cpu numpy
pip install numpy==1.26.4
pip install faiss-cpu==1.9.0 --no-cache-dir

# Clear Python cache
echo "🧹 Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo "✅ Done! Try running the app again."
echo "Run: streamlit run app.py"
