from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import joblib

from rag_engine import (
    build_rag_knowledge,
    build_short_reasons,
    build_llm_context,
    get_short_rag_point
)
from llm_engine import generate_short_llm_insights


app = FastAPI(
    title="ArchPilot AI",
    description="ML, RAG, Validation and Local LLM-based Architecture Recommendation System",
    version="14.0"
)

bundle = joblib.load("archpilot_model.pkl")

models = bundle["models"]
feature_encoders = bundle["feature_encoders"]
target_encoders = bundle["target_encoders"]
feature_columns = bundle["feature_columns"]
model_algorithm = bundle["model_algorithm"]


class AppScenario(BaseModel):
    users_count: int
    traffic_level: Literal["Low", "Medium", "High"]
    team_size: int
    update_frequency: Literal["Monthly", "Weekly", "Daily"]
    integrations: int
    security_level: Literal["Low", "Medium", "High"]
    scalability_need: Literal["Low", "Medium", "High"]


@app.get("/")
def home():
    return {
        "project": "ArchPilot AI",
        "message": "Architecture + Deployment + API recommendation system"
    }


@app.post("/predict")
def predict_architecture(data: AppScenario):
    encoded_input = {
        "users_count": data.users_count,
        "traffic_level": feature_encoders["traffic_level"].transform([data.traffic_level])[0],
        "team_size": data.team_size,
        "update_frequency": feature_encoders["update_frequency"].transform([data.update_frequency])[0],
        "integrations": data.integrations,
        "security_level": feature_encoders["security_level"].transform([data.security_level])[0],
        "scalability_need": feature_encoders["scalability_need"].transform([data.scalability_need])[0],
    }

    input_df = pd.DataFrame([encoded_input], columns=feature_columns)

    architecture_encoded = models["architecture"].predict(input_df)[0]
    deployment_encoded = models["deployment_method"].predict(input_df)[0]
    api_encoded = models["api_style"].predict(input_df)[0]

    architecture = target_encoders["architecture"].inverse_transform([architecture_encoded])[0]
    deployment_method = target_encoders["deployment_method"].inverse_transform([deployment_encoded])[0]
    api_style = target_encoders["api_style"].inverse_transform([api_encoded])[0]

    # Validation rules are applied internally.
    architecture, deployment_method, api_style = apply_validation_rules(
        data=data,
        architecture=architecture,
        deployment_method=deployment_method,
        api_style=api_style
    )

    # RAG is used internally for LLM support.
    rag_knowledge = build_rag_knowledge(
        input_data=data,
        architecture=architecture,
        deployment_method=deployment_method,
        api_style=api_style
    )

    why_predicted = build_short_reasons(
        input_data=data,
        architecture=architecture,
        deployment_method=deployment_method
    )

    llm_context = build_llm_context(
        input_data=data,
        architecture=architecture,
        deployment_method=deployment_method,
        api_style=api_style,
        rag_knowledge=rag_knowledge
    )

    llm_insights = generate_short_llm_insights(
        input_data=data,
        architecture=architecture,
        deployment_method=deployment_method,
        api_style=api_style,
        llm_context=llm_context
    )

    return {
        "architecture_suggestion": {
            "recommended_architecture": architecture,
            "recommended_deployment_method": deployment_method,
            "recommended_api_style": api_style,
            "model_algorithm_used": model_algorithm
        },

        "authentication": "JWT Token Authentication can secure API access.",

        "authorization": "RBAC can control user roles and permissions.",

        "deployment_suggestion": get_deployment_suggestion(deployment_method),

        "why_model_predicted_this": why_predicted,

        "why_not_other_methods": get_why_not_others(architecture),

        "llm_generated_short_insights": llm_insights
    }


def apply_validation_rules(data, architecture, deployment_method, api_style):
    """
    Validation layer improves obvious architecture decisions.
    It works internally but is not displayed in the API output.
    """

    # Small/simple application
    if (
        data.users_count <= 5000
        and data.traffic_level == "Low"
        and data.team_size <= 5
        and data.update_frequency == "Monthly"
        and data.integrations <= 3
        and data.scalability_need == "Low"
    ):
        return (
            "Monolithic",
            "Single Server Deployment",
            "REST API"
        )

    # Medium application
    if (
        5000 < data.users_count <= 20000
        and data.traffic_level in ["Low", "Medium"]
        and data.team_size <= 15
        and data.integrations <= 8
        and data.scalability_need in ["Low", "Medium"]
    ):
        return (
            "Containerized App",
            "Docker Container Deployment",
            "REST API"
        )

    # Large modular application
    if (
        20000 < data.users_count <= 50000
        and data.team_size > 15
        and data.integrations > 8
        and data.traffic_level in ["Medium", "High"]
        and data.scalability_need in ["Medium", "High"]
    ):
        return (
            "Microservices",
            "Microservices with Docker",
            "REST API + gRPC"
        )

    # High-scale production application
    if (
        data.users_count > 50000
        and data.traffic_level == "High"
        and data.update_frequency == "Daily"
        and data.integrations > 8
        and data.security_level == "High"
        and data.scalability_need == "High"
    ):
        return (
            "Kubernetes Deployment",
            "Kubernetes Orchestration Deployment",
            "REST API + gRPC/Event-Driven"
        )

    # If no validation rule matches, use ML output.
    return architecture, deployment_method, api_style


def get_deployment_suggestion(deployment_method):
    if "Kubernetes" in deployment_method:
        return get_short_rag_point("Kubernetes Deployment")

    if "Microservices" in deployment_method:
        return "Use Docker to package and deploy each microservice independently."

    if "Docker" in deployment_method:
        return get_short_rag_point("Docker Deployment")

    if "Single Server" in deployment_method:
        return "Use simple single-server deployment for small low-traffic applications."

    return get_short_rag_point("Cloud PaaS")


def get_why_not_others(architecture):
    if architecture == "Kubernetes Deployment":
        return [
            "Monolithic cannot easily handle high traffic and scaling.",
            "Docker alone lacks orchestration, self-healing, and auto-scaling."
        ]

    if architecture == "Microservices":
        return [
            "Monolithic can become hard to maintain at scale.",
            "Kubernetes may add extra complexity if orchestration is not required."
        ]

    if architecture == "Containerized App":
        return [
            "Plain server deployment may cause environment mismatch.",
            "Kubernetes may be excessive for medium workloads."
        ]

    return [
        "Microservices adds unnecessary complexity for small systems.",
        "Kubernetes may increase cost and operational effort."
    ]