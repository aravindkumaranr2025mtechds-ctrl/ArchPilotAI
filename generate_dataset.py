import random
import pandas as pd


random.seed(42)


def create_monolithic_rows(count):
    rows = []

    for _ in range(count):
        rows.append({
            "users_count": random.randint(100, 5000),
            "traffic_level": "Low",
            "team_size": random.randint(2, 5),
            "update_frequency": "Monthly",
            "integrations": random.randint(1, 3),
            "security_level": random.choice(["Low", "Medium"]),
            "scalability_need": "Low",
            "architecture": "Monolithic",
            "deployment_method": "Single Server Deployment",
            "api_style": "REST API"
        })

    return rows


def create_containerized_rows(count):
    rows = []

    for _ in range(count):
        rows.append({
            "users_count": random.randint(5001, 20000),
            "traffic_level": random.choice(["Low", "Medium"]),
            "team_size": random.randint(6, 15),
            "update_frequency": random.choice(["Monthly", "Weekly"]),
            "integrations": random.randint(3, 8),
            "security_level": random.choice(["Medium", "High"]),
            "scalability_need": random.choice(["Low", "Medium"]),
            "architecture": "Containerized App",
            "deployment_method": "Docker Container Deployment",
            "api_style": "REST API"
        })

    return rows


def create_microservices_rows(count):
    rows = []

    for _ in range(count):
        rows.append({
            "users_count": random.randint(20001, 50000),
            "traffic_level": random.choice(["Medium", "High"]),
            "team_size": random.randint(16, 30),
            "update_frequency": random.choice(["Weekly", "Daily"]),
            "integrations": random.randint(9, 15),
            "security_level": random.choice(["Medium", "High"]),
            "scalability_need": random.choice(["Medium", "High"]),
            "architecture": "Microservices",
            "deployment_method": "Microservices with Docker",
            "api_style": "REST API + gRPC"
        })

    return rows


def create_kubernetes_rows(count):
    rows = []

    for _ in range(count):
        rows.append({
            "users_count": random.randint(50001, 200000),
            "traffic_level": "High",
            "team_size": random.randint(20, 50),
            "update_frequency": "Daily",
            "integrations": random.randint(10, 25),
            "security_level": "High",
            "scalability_need": "High",
            "architecture": "Kubernetes Deployment",
            "deployment_method": "Kubernetes Orchestration Deployment",
            "api_style": "REST API + gRPC/Event-Driven"
        })

    return rows


def add_boundary_cases():
    """
    These are important examples to make sure the model learns expected demo scenarios.
    """

    return [
        {
            "users_count": 1200,
            "traffic_level": "Low",
            "team_size": 4,
            "update_frequency": "Monthly",
            "integrations": 2,
            "security_level": "Low",
            "scalability_need": "Low",
            "architecture": "Monolithic",
            "deployment_method": "Single Server Deployment",
            "api_style": "REST API"
        },
        {
            "users_count": 900,
            "traffic_level": "Low",
            "team_size": 3,
            "update_frequency": "Monthly",
            "integrations": 1,
            "security_level": "Low",
            "scalability_need": "Low",
            "architecture": "Monolithic",
            "deployment_method": "Single Server Deployment",
            "api_style": "REST API"
        },
        {
            "users_count": 12000,
            "traffic_level": "Medium",
            "team_size": 9,
            "update_frequency": "Weekly",
            "integrations": 5,
            "security_level": "Medium",
            "scalability_need": "Medium",
            "architecture": "Containerized App",
            "deployment_method": "Docker Container Deployment",
            "api_style": "REST API"
        },
        {
            "users_count": 28000,
            "traffic_level": "Medium",
            "team_size": 18,
            "update_frequency": "Weekly",
            "integrations": 9,
            "security_level": "Medium",
            "scalability_need": "High",
            "architecture": "Microservices",
            "deployment_method": "Microservices with Docker",
            "api_style": "REST API + gRPC"
        },
        {
            "users_count": 85000,
            "traffic_level": "High",
            "team_size": 28,
            "update_frequency": "Daily",
            "integrations": 16,
            "security_level": "High",
            "scalability_need": "High",
            "architecture": "Kubernetes Deployment",
            "deployment_method": "Kubernetes Orchestration Deployment",
            "api_style": "REST API + gRPC/Event-Driven"
        }
    ]


def generate_dataset():
    data = []

    data.extend(create_monolithic_rows(2000))
    data.extend(create_containerized_rows(2000))
    data.extend(create_microservices_rows(2000))
    data.extend(create_kubernetes_rows(2000))
    data.extend(add_boundary_cases())

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv("dataset.csv", index=False)

    print("dataset.csv created successfully!")
    print("Total rows:", len(df))

    print("\nArchitecture distribution:")
    print(df["architecture"].value_counts())

    print("\nDeployment method distribution:")
    print(df["deployment_method"].value_counts())

    print("\nAPI style distribution:")
    print(df["api_style"].value_counts())


if __name__ == "__main__":
    generate_dataset()