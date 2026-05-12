import os
import requests


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate"
)

MODEL_NAME = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2:3b"
)

def limit_words(text, max_words=18):
    words = text.replace("\n", " ").split()
    return " ".join(words[:max_words])


def generate_fallback_insights(input_data, architecture, deployment_method, api_style):
    why_recommended = []
    deployment_reason = []
    advantages = []
    why_not_other_methods = []

    if architecture == "Microservices":
        why_recommended = [
            f"Microservices fit {input_data.team_size} developers, {input_data.integrations} integrations, and {input_data.scalability_need} scalability.",
            "Services can be developed, deployed, and scaled independently."
        ]

        deployment_reason = [
            "Docker packages each service and keeps deployment consistent across environments."
        ]

        advantages = [
            "Independent service development improves maintainability.",
            "Service-level scaling supports future growth.",
            "REST and gRPC support external and internal communication."
        ]

        why_not_other_methods = [
            "Monolithic is harder to maintain when modules and teams grow.",
            "Kubernetes can be added later when traffic becomes high."
        ]

    elif architecture == "Kubernetes Deployment":
        why_recommended = [
            "Kubernetes fits high users, high traffic, frequent updates, and high scalability.",
            "It supports orchestration, self-healing, and production-grade deployment."
        ]

        deployment_reason = [
            "Kubernetes manages containers with auto-scaling, load balancing, and rolling updates."
        ]

        advantages = [
            "Auto-scaling handles high traffic.",
            "Load balancing improves availability.",
            "Self-healing restarts failed services."
        ]

        why_not_other_methods = [
            "Monolithic cannot easily handle high traffic and scaling.",
            "Docker alone lacks orchestration and self-healing."
        ]

    elif architecture == "Containerized App":
        why_recommended = [
            "Containerized deployment fits medium traffic and moderate scalability.",
            "Docker avoids environment mismatch across development and production."
        ]

        deployment_reason = [
            "Docker packages application dependencies for consistent deployment."
        ]

        advantages = [
            "Portable deployment across environments.",
            "Faster release and easier packaging.",
            "Reduced dependency mismatch issues."
        ]

        why_not_other_methods = [
            "Plain server deployment may cause environment mismatch.",
            "Kubernetes may be excessive for medium workloads."
        ]

    else:
        why_recommended = [
            "Monolithic fits small teams, low traffic, and simple applications.",
            "It reduces development and deployment complexity."
        ]

        deployment_reason = [
            "Single server deployment is simple and low-cost for small applications."
        ]

        advantages = [
            "Simple development and testing.",
            "Low operational complexity.",
            "Easy deployment for small teams."
        ]

        why_not_other_methods = [
            "Microservices adds unnecessary complexity for small systems.",
            "Kubernetes increases cost and operational effort."
        ]

    return {
        "why_recommended": [limit_words(x) for x in why_recommended],
        "deployment_reason": [limit_words(x) for x in deployment_reason],
        "advantages": [limit_words(x) for x in advantages],
        "why_not_other_methods": [limit_words(x) for x in why_not_other_methods]
    }


def parse_llm_text_to_sections(text, fallback):
    sections = {
        "why_recommended": [],
        "deployment_reason": [],
        "advantages": [],
        "why_not_other_methods": []
    }

    current_section = None

    for raw_line in text.split("\n"):
        line = raw_line.strip()

        if not line:
            continue

        lower = line.lower()

        if lower.startswith("why recommended"):
            current_section = "why_recommended"
            continue

        if lower.startswith("deployment reason"):
            current_section = "deployment_reason"
            continue

        if lower.startswith("advantages"):
            current_section = "advantages"
            continue

        if lower.startswith("why not"):
            current_section = "why_not_other_methods"
            continue

        if line.startswith("-") and current_section:
            clean_line = line.replace("-", "").strip()

            if "singlecontainer" in clean_line.lower():
                continue

            sections[current_section].append(limit_words(clean_line))

    if not sections["why_recommended"] or not sections["advantages"]:
        return fallback

    sections["why_recommended"] = sections["why_recommended"][:2]
    sections["deployment_reason"] = sections["deployment_reason"][:1]
    sections["advantages"] = sections["advantages"][:3]
    sections["why_not_other_methods"] = sections["why_not_other_methods"][:2]

    return sections


def generate_short_llm_insights(input_data, architecture, deployment_method, api_style, llm_context):
    fallback = generate_fallback_insights(
        input_data=input_data,
        architecture=architecture,
        deployment_method=deployment_method,
        api_style=api_style
    )

    prompt = f"""
You are an industry software architect.

Do NOT write paragraphs.
Each bullet must be maximum 18 words.
Do NOT change architecture, deployment method, or API style.
Do NOT use the word SingleContainer.

Architecture: {architecture}
Deployment Method: {deployment_method}
API Style: {api_style}

User Inputs:
users_count={input_data.users_count}
traffic_level={input_data.traffic_level}
team_size={input_data.team_size}
update_frequency={input_data.update_frequency}
integrations={input_data.integrations}
security_level={input_data.security_level}
scalability_need={input_data.scalability_need}

Important logic:
- If architecture is Microservices with Docker, deployment reason must mention Docker, not Kubernetes.
- If traffic is Medium, say Kubernetes can be added later, not required immediately.
- If architecture is Kubernetes Deployment, mention Kubernetes orchestration.
- If architecture is Monolithic, mention simplicity and low complexity.
- If architecture is Containerized App, mention Docker consistency and portability.

Return only this format:

Why recommended:
- <short reason>
- <short reason>

Deployment reason:
- <short reason>

Advantages:
- <advantage>
- <advantage>
- <advantage>

Why not other methods:
- <why weaker method is avoided>
- <why weaker method is avoided>
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 180
                }
            },
            timeout=180
        )

        response.raise_for_status()
        result = response.json()
        raw_output = result.get("response", "")

        parsed = parse_llm_text_to_sections(raw_output, fallback)

        return parsed

    except requests.exceptions.ConnectionError:
        return fallback

    except requests.exceptions.Timeout:
        return fallback

    except Exception:
        return fallback