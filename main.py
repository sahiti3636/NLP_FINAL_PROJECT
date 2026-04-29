"""
main.py
=======
Dreamscape Mapper — FastAPI Backend
Wraps dream_pipeline_p.py (production pipeline) and exposes POST /analyze
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dream_pipeline_p import DreamPipelineModel, run_production_pipeline
import json

# ---------------------------------------------------------------------------
# Load model once at startup (heavy — BiLSTM + embeddings)
# ---------------------------------------------------------------------------
try:
    _MODEL = DreamPipelineModel()
except Exception as _e:
    _MODEL = None
    print(f"WARNING: Could not load production model: {_e}")

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
    Run the production BiLSTM dream pipeline on the supplied text.
    Returns a structured thematic-emotional JSON record.
    """
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Production model failed to load at startup.")

    dream_text = payload.dream_text.strip()
    if not dream_text:
        raise HTTPException(status_code=422, detail="dream_text must not be empty.")

    try:
        result = run_production_pipeline(dream_text, _MODEL)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    # Keep only the 8 core NRC emotions (strip valence if present)
    core_emotions = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
    emotion_vector = {
        k: round(result["emotion_vector"].get(k, 0.0), 4)
        for k in core_emotions
    }

    # dream_pipeline_p uses lowercase keys for semantic relations: agent/action/target
    semantic_relations = result.get("semantic_relations", [])

    return AnalysisResponse(
        Topic_Cluster=result["topic_cluster"],
        Dominant_Emotion=result["dominant_emotion"],
        Key_Entities=result["key_entities"],
        Semantic_Relation=[
            SemanticRelation(
                Agent=s.get("agent") or "Unknown",
                Action=s.get("action") or "Unknown",
                Target=s.get("target") or "Unknown",
            )
            for s in semantic_relations
        ],
        Emotion_Vector=emotion_vector,
        Global_Stat=result.get("summary", ""),
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
