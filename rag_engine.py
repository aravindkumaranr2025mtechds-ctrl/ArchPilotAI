MAX_WORDS = 20


def limit_words(text, max_words=MAX_WORDS):
    words = text.replace("\n", " ").split()
    return " ".join(words[:max_words])


def load_knowledge_base(file_path="knowledge_base.txt"):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
    return chunks


def extract_topic(chunk):
    first_line = chunk.split("\n")[0].strip()
    return first_line.replace(":", "")


def extract_insight(chunk):
    for line in chunk.split("\n"):
        if line.lower().startswith("insight:"):
            return line.replace("insight:", "").strip()
    return chunk


def get_knowledge_by_topic(topic_name):
    """
    Retrieves exact topic from knowledge_base.txt.
    This avoids wrong similarity matches like team_size=30 matching 'Why Not Microservices'.
    """

    chunks = load_knowledge_base()

    for chunk in chunks:
        topic = extract_topic(chunk)
        if topic.lower() == topic_name.lower():
            return {
                "matched_topic": topic,
                "matched_knowledge": limit_words(extract_insight(chunk))
            }

    return {
        "matched_topic": topic_name,
        "matched_knowledge": "No exact knowledge found for this input."
    }


def choose_user_topic(input_data, feature_name):
    if feature_name == "users_count":
        if input_data.users_count > 50000:
            return "High Users"
        elif input_data.users_count > 5000:
            return "Medium Users"
        return "Monolithic Architecture"

    if feature_name == "traffic_level":
        if input_data.traffic_level == "High":
            return "High Traffic"
        elif input_data.traffic_level == "Medium":
            return "Containerized App"
        return "Monolithic Architecture"

    if feature_name == "team_size":
        if input_data.team_size > 15:
            return "Large Team"
        elif input_data.team_size > 5:
            return "Medium Team"
        return "Small Team"

    if feature_name == "update_frequency":
        if input_data.update_frequency == "Daily":
            return "Daily Updates"
        elif input_data.update_frequency == "Weekly":
            return "Containerized App"
        return "Monolithic Architecture"

    if feature_name == "integrations":
        if input_data.integrations > 8:
            return "Many Integrations"
        elif input_data.integrations > 3:
            return "Medium Integrations"
        return "Few Integrations"

    if feature_name == "security_level":
        if input_data.security_level == "High":
            return "High Security"
        elif input_data.security_level == "Medium":
            return "Medium Security"
        return "JWT Authentication"

    if feature_name == "scalability_need":
        if input_data.scalability_need == "High":
            return "High Scalability"
        elif input_data.scalability_need == "Medium":
            return "Medium Scalability"
        return "Monolithic Architecture"

    return "Decision Support"


def build_rag_knowledge(input_data, architecture, deployment_method, api_style):
    """
    Builds short, controlled RAG insights for each user input.
    Every matched_knowledge is limited to 20 words.
    """

    feature_values = [
        ("users_count", f"users_count = {input_data.users_count}"),
        ("traffic_level", f"traffic_level = {input_data.traffic_level}"),
        ("team_size", f"team_size = {input_data.team_size}"),
        ("update_frequency", f"update_frequency = {input_data.update_frequency}"),
        ("integrations", f"integrations = {input_data.integrations}"),
        ("security_level", f"security_level = {input_data.security_level}"),
        ("scalability_need", f"scalability_need = {input_data.scalability_need}")
    ]

    rag_output = []

    for feature_name, user_input in feature_values:
        topic = choose_user_topic(input_data, feature_name)
        knowledge = get_knowledge_by_topic(topic)

        rag_output.append({
            "user_input": user_input,
            "matched_topic": knowledge["matched_topic"],
            "matched_knowledge": knowledge["matched_knowledge"]
        })

    architecture_topic = architecture
    if architecture == "Kubernetes Deployment":
        architecture_topic = "Kubernetes Deployment"
    elif architecture == "Microservices":
        architecture_topic = "Microservices Architecture"
    elif architecture == "Containerized App":
        architecture_topic = "Containerized App"
    elif architecture == "Monolithic":
        architecture_topic = "Monolithic Architecture"

    architecture_knowledge = get_knowledge_by_topic(architecture_topic)

    rag_output.append({
        "user_input": f"architecture = {architecture}",
        "matched_topic": architecture_knowledge["matched_topic"],
        "matched_knowledge": architecture_knowledge["matched_knowledge"]
    })

    deployment_knowledge = get_knowledge_by_topic(get_deployment_topic(deployment_method))

    rag_output.append({
        "user_input": f"deployment_method = {deployment_method}",
        "matched_topic": deployment_knowledge["matched_topic"],
        "matched_knowledge": deployment_knowledge["matched_knowledge"]
    })

    api_knowledge = get_knowledge_by_topic(get_api_topic(api_style))

    rag_output.append({
        "user_input": f"api_style = {api_style}",
        "matched_topic": api_knowledge["matched_topic"],
        "matched_knowledge": api_knowledge["matched_knowledge"]
    })

    return rag_output


def get_deployment_topic(deployment_method):
    if "Kubernetes" in deployment_method:
        return "Kubernetes Deployment"
    if "Docker" in deployment_method:
        return "Docker Deployment"
    if "Single Server" in deployment_method:
        return "Monolithic Architecture"
    return "Cloud PaaS"


def get_api_topic(api_style):
    if "gRPC" in api_style:
        return "gRPC API"
    if "Event" in api_style:
        return "Event Driven"
    return "REST API"


def build_short_reasons(input_data, architecture, deployment_method):
    reasons = []

    if input_data.users_count > 50000:
        reasons.append("Very large users need scalable deployment.")

    if input_data.traffic_level == "High":
        reasons.append("High traffic needs load balancing.")

    if input_data.team_size > 15:
        reasons.append("Large team benefits from modular services.")

    if input_data.update_frequency == "Daily":
        reasons.append("Daily updates need rolling deployment.")

    if input_data.integrations > 8:
        reasons.append("Many integrations need service separation.")

    if input_data.security_level == "High":
        reasons.append("High security needs JWT and RBAC.")

    if input_data.scalability_need == "High":
        reasons.append(" High scalability needs independent service scaling.")

    if not reasons:
        reasons.append(f"Input pattern matches {architecture} with {deployment_method}.")

    return [limit_words(reason) for reason in reasons]


def get_short_rag_point(topic_name):
    result = get_knowledge_by_topic(topic_name)
    return result["matched_knowledge"]


def unique_list(items):
    unique = []
    for item in items:
        if item not in unique:
            unique.append(item)
    return unique


def build_llm_context(input_data, architecture, deployment_method, api_style, rag_knowledge):
    return {
        "architecture": architecture,
        "deployment_method": deployment_method,
        "api_style": api_style,
        "user_inputs": {
            "users_count": input_data.users_count,
            "traffic_level": input_data.traffic_level,
            "team_size": input_data.team_size,
            "update_frequency": input_data.update_frequency,
            "integrations": input_data.integrations,
            "security_level": input_data.security_level,
            "scalability_need": input_data.scalability_need
        },
        "rag_knowledge": rag_knowledge
    }