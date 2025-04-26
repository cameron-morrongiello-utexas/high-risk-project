import json
import random

ages = list(range(1, 90))
genders = ["male", "female", "non-binary"]
symptoms = [
    "chest pain and shortness of breath",
    "severe headache and blurred vision",
    "sore throat and mild fever",
    "abdominal cramps and diarrhea",
    "persistent cough and fatigue",
    "rash on arms and itching",
    "swollen ankle after twisting it",
    "frequent urination and excessive thirst",
    "runny nose and sneezing",
    "dizziness and feeling faint"
]
times = ["just now", "30 minutes ago", "2 hours ago", "yesterday", "2 days ago", "last week"]
histories = [
    "No significant medical history.",
    "Has a history of asthma.",
    "Diabetic and takes insulin.",
    "History of heart disease.",
    "Currently pregnant.",
    "No known chronic conditions."
]
triage_levels = ["Emergency Room", "Urgent Care", "Primary Care", "Self-Care"]

triage_mapping = {
    "chest pain": ("Emergency Room", "Chest pain could indicate a heart attack or other serious cardiac issues."),
    "severe headache": ("Urgent Care", "A sudden severe headache might be a sign of a neurological emergency."),
    "sore throat": ("Self-Care", "A sore throat and mild fever are typically self-limiting and do not require urgent care."),
    "abdominal cramps": ("Urgent Care", "Persistent abdominal pain can be a sign of infection or digestive issues needing medical attention."),
    "persistent cough": ("Primary Care", "A cough with fatigue might indicate an underlying respiratory condition."),
    "rash": ("Self-Care", "Skin rashes without other symptoms are usually manageable at home."),
    "swollen ankle": ("Urgent Care", "Swelling after twisting the ankle might indicate a sprain, often handled in urgent care."),
    "frequent urination": ("Primary Care", "Frequent urination and thirst can be symptoms of diabetes, requiring primary care follow-up."),
    "runny nose": ("Self-Care", "Runny nose and sneezing suggest a common cold, which resolves with self-care."),
    "dizziness": ("Emergency Room", "Sudden dizziness or fainting may signal serious issues like stroke or cardiac dysfunction.")
}

def generate_case():
    age = random.choice(ages)
    gender = random.choice(genders)
    symptom = random.choice(symptoms)
    time = random.choice(times)
    history = random.choice(histories)

    # Base decision based on symptom
    triage = "Primary Care"
    rationale = "Based on the symptoms and history, the case is best managed by a primary care provider."
    
    for keyword in triage_mapping:
        if keyword in symptom:
            triage, rationale = triage_mapping[keyword]
            break

    # Secondary decision based on time of onset
    if "just now" in time or "30 minutes ago" in time:
        triage = "Emergency Room" if triage == "Primary Care" else triage
        rationale += " Symptoms have just started, requiring more immediate attention."

    if "yesterday" in time or "2 days ago" in time:
        triage = "Primary Care" if triage == "Urgent Care" else triage
        rationale += " Symptoms have been present for a while and are less urgent."

    # Secondary decision based on medical history
    if "heart disease" in history and "chest pain" in symptom:
        triage = "Emergency Room"
        rationale += " Given the patient's history of heart disease, chest pain must be addressed immediately."
    
    if "diabetic" in history and "frequent urination" in symptom:
        triage = "Primary Care"
        rationale += " The patient’s diabetes requires ongoing management for symptoms like frequent urination."

    # Construct the description and output
    description = f"A {age}-year-old {gender} reports {symptom}. Symptoms started {time}. {history}"
    output = f"{rationale} Therefore, the appropriate triage level is: {triage}."

    return {
        "input": description,
        "output": output
    }

filename = "data/valid.jsonl"
with open(filename, "w") as f:
    for _ in range(1000):
        case = generate_case()
        f.write(json.dumps(case) + "\n")
