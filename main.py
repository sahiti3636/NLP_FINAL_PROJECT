"""
main.py
=======
Dreamscape Mapper — FastAPI Backend
Wraps dream_pipeline.py and exposes POST /analyze
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dream_pipeline import run_pipeline
import json

app = FastAPI(
    title="Dreamscape Mapper API",
    description="NRC-based dream analysis pipeline",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS — allow all origins for local React development
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class DreamRequest(BaseModel):
    dream_text: str


class SemanticRelation(BaseModel):
    Agent: str
    Action: str
    Target: str


class AnalysisResponse(BaseModel):
    Topic_Cluster: str
    Dominant_Emotion: str
    Key_Entities: list[str]
    Semantic_Relation: list[SemanticRelation]
    Emotion_Vector: dict[str, float]
    Global_Stat: str


class Theme(BaseModel):
    topic_label: str
    size: int


class GlobalInsightsResponse(BaseModel):
    total_dreams: int
    unique_archetypes: int
    semantic_density: float
    dominant_tone: str
    emotion_radar: dict[str, float]
    top_themes: list[Theme]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok", "service": "Dreamscape Mapper API"}


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_dream(payload: DreamRequest):
    """
    Run the 11-step NRC-based dream pipeline on the supplied text.
    Returns a structured thematic-emotional JSON record.
    """
    dream_text = payload.dream_text.strip()
    if not dream_text:
        raise HTTPException(status_code=422, detail="dream_text must not be empty.")

    try:
        result = run_pipeline(dream_text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")

    # Strip valence dimensions (positive / negative) — keep only the 8 core NRC emotions
    core_emotions = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
    emotion_vector = {
        k: round(result["Emotion_Vector"].get(k, 0.0), 4)
        for k in core_emotions
    }

    return AnalysisResponse(
        Topic_Cluster=result["Topic_Cluster"],
        Dominant_Emotion=result["Dominant_Emotion"],
        Key_Entities=result["Key_Entities"],
        Semantic_Relation=[
            SemanticRelation(Agent=s["Agent"], Action=s["Action"], Target=s["Target"])
            for s in result["Semantic_Relation"]
        ],
        Emotion_Vector=emotion_vector,
        Global_Stat=result["Global_Stat"],
    )


@app.get("/global-insights", response_model=GlobalInsightsResponse)
def get_global_insights():
    """
    Return global statistics to populate the Global Insights dashboard.
    """
    try:
        with open('jsons/step9_results.json', 'r') as f:
            step9 = json.load(f)
        with open('jsons/step11_global_statistics.json', 'r') as f:
            step11 = json.load(f)
        with open('jsons/dream_annotations.json', 'r') as f:
            annotations = json.load(f)
        
        total_dreams = len(annotations)
        unique_archetypes = len(set(x["topic_label"] for x in step9 if "topic_label" in x))
        
        total_keywords = sum(len(x.get("keywords", [])) for x in step9)
        semantic_density = total_keywords / max(len(step9), 1)
        
        global_radar = step11.get("global_affective_averages", {})
        
        dominant_tone = max(global_radar.items(), key=lambda k: k[1])[0] if global_radar else "Unknown"
        dominant_tone = dominant_tone.capitalize()
        
        # Sort top themes by size
        sorted_themes = sorted(step9, key=lambda x: x.get("size", 0), reverse=True)
        top_10 = sorted_themes[:10]
        
        return GlobalInsightsResponse(
            total_dreams=total_dreams,
            unique_archetypes=unique_archetypes,
            semantic_density=round(semantic_density, 2),
            dominant_tone=dominant_tone,
            emotion_radar=global_radar,
            top_themes=[Theme(topic_label=t.get("topic_label", "Unknown"), size=t.get("size", 0)) for t in top_10]
        )
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Data loading error: {str(exc)}")
