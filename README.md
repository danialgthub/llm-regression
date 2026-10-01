# Smart Resume Analyzer & Career Recommender

This project evaluates an email classifier against a golden dataset, generates reports, and tracks accuracy trends over time.  
It supports both **Mock Mode** (free, no API key) and **Real Mode** (requires OpenAI API key).

---

## 🚀 Quick Start

### 1. Build the Docker image
```bash
docker build -t eval-runner .

### 2. Run in Mock Mode (free, no API key)
docker run -e MOCK_MODE=true eval-runner

### 3. Run in Real Mode (requires API key)
docker run -e OPENAI_API_KEY=sk-yourkeyhere eval-runner
