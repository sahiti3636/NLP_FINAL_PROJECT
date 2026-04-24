# NLP Final Project: Dreamscape Mapper

This repository contains the "Dreamscape Mapper", a cinematic, full-stack application that analyzes and maps dreams using a comprehensive NLP pipeline. It features a robust Python FastAPI backend and a highly polished React (Vite + TypeScript + TailwindCSS) frontend.

## Features

- **Dream Chamber**: A distraction-free, beautifully composed input interface for submitting dreams.
- **NLP Analysis Pipeline**: Deep analysis of dream texts to extract themes, sentiment, and sematic relations.
- **Glassmorphic Results Dashboard**: An aesthetic and fully responsive results view.
- **Emotion Radar View**: Real-time rendering of NRC emotion charts based on NLP output.

## Architecture

- **Backend**: Python, FastAPI, and specialized NLP logic inside `dream_pipeline.py` and `main.py`.
- **Frontend**: React.js with Vite, styled with Tailwind CSS and Framer Motion for micro-animations.

## Getting Started

### Backend
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend
1. Navigate to the `frontend/` directory.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
