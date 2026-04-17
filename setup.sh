#!/usr/bin/env bash
# ============================================================
# HR Copilot — Setup Script (Mac / Linux)
# chmod +x setup.sh && ./setup.sh
# ============================================================
set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

echo -e "${BLUE}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║  HR Copilot — Multi-Agent Setup                           ║"
echo "  ║  Maneesh Kumar                                            ║"
echo "  ╚══════════════════════════════════════════════════════════╝${NC}"

# 1. Python check
echo -e "\n${CYAN}[1/7] Python version check...${NC}"
python3 -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" \
  && echo -e "${GREEN}  ✅ Python $(python3 --version | awk '{print $2}')${NC}" \
  || { echo -e "${RED}  ❌ Python 3.10+ required. Install from python.org${NC}"; exit 1; }

# 2. Virtual environment
echo -e "\n${CYAN}[2/7] Creating virtual environment...${NC}"
[ -d ".venv" ] && echo -e "${YELLOW}  .venv already exists — skipping${NC}" \
               || python3 -m venv .venv
source .venv/bin/activate
echo -e "${GREEN}  ✅ $(which python3)${NC}"

# 3. pip upgrade
echo -e "\n${CYAN}[3/7] Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}  ✅ pip ready${NC}"

# 4. Install dependencies
echo -e "\n${CYAN}[4/7] Installing Python dependencies (3-5 min first time)...${NC}"
pip install -r requirements.txt --quiet
echo -e "${GREEN}  ✅ All dependencies installed (including Streamlit)${NC}"

# 5. Download AI models
echo -e "\n${CYAN}[5/7] Downloading AI models...${NC}"
python3 - << 'PYEOF'
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

print("  Downloading all-MiniLM-L6-v2 (embedding model, ~90MB)...")
SentenceTransformer("all-MiniLM-L6-v2")
print("  ✅ Embedding model ready")

print("  Downloading ms-marco cross-encoder (reranker, ~80MB)...")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
print("  ✅ Reranker ready")

print("  Downloading nli-deberta-v3-small (compliance guard, ~180MB)...")
pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small")
print("  ✅ NLI model ready")
PYEOF

# 6. Create sample HR data and build index
echo -e "\n${CYAN}[6/7] Creating HR sample data + building knowledge index...${NC}"
python3 create_sample_data.py
echo -e "${GREEN}  ✅ HR documents and structured data created${NC}"

echo -e "${YELLOW}  Building FAISS + BM25 index from HR documents...${NC}"
python3 component_a_hr_indexing.py
echo -e "${GREEN}  ✅ Knowledge index built${NC}"

# 7. Ollama install + Mistral pull
echo -e "\n${CYAN}[7/7] Ollama + Mistral 7B setup...${NC}"
if command -v ollama &>/dev/null; then
    echo -e "${GREEN}  ✅ Ollama already installed${NC}"
else
    echo -e "${YELLOW}  Installing Ollama...${NC}"
    if [[ "$(uname)" == "Darwin" ]]; then
        brew install ollama 2>/dev/null || curl -fsSL https://ollama.com/install.sh | sh
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    if command -v ollama &>/dev/null; then
        echo -e "${GREEN}  ✅ Ollama installed${NC}"
    else
        echo -e "${RED}  ❌ Ollama install failed. Install manually from https://ollama.com${NC}"
        echo -e "${RED}     Then run: ollama pull mistral${NC}"
    fi
fi

if command -v ollama &>/dev/null; then
    echo -e "${YELLOW}  Starting Ollama server...${NC}"
    ollama serve >/tmp/ollama_hr.log 2>&1 & sleep 3
    echo -e "${YELLOW}  Pulling Mistral 7B (~4.1 GB, first time only)...${NC}"
    ollama pull mistral && echo -e "${GREEN}  ✅ Mistral 7B ready${NC}" \
                        || echo -e "${RED}  ❌ Mistral pull failed — check your internet connection${NC}"
fi

echo -e "\n${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}  ✅  SETUP COMPLETE!${NC}"
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  Activate environment:  ${YELLOW}source .venv/bin/activate${NC}"
echo ""
echo -e "  ${BOLD}Launch the Streamlit app:${NC}"
echo -e "  ${CYAN}streamlit run hr_copilot_ui.py${NC}  →  http://localhost:8501"
echo ""
echo -e "  Or run phases individually:"
echo -e "  ${CYAN}python3 component_a_hr_indexing.py${NC}           Phase 1"
echo -e "  ${CYAN}python3 component_b_orchestrator_agent.py${NC}    Phase 2"
echo -e "  ${CYAN}python3 component_c_policy_data_agents.py${NC}    Phase 3"
echo -e "  ${CYAN}python3 component_d_compliance_guard.py${NC}      Phase 4"
echo -e "  ${CYAN}python3 component_e_response_synthesizer.py${NC}  Phase 5"
echo -e "  ${CYAN}python3 hr_copilot_pipeline.py${NC}               Phase 6 (CLI)"
echo ""
