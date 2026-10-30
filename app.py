
import os
import gradio as gr
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Load API key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing from your .env file.")

# Create AI model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.4,
    api_key=api_key
)


# Study plan generator function
def generate_study_plan(subjects, hours, exam_dates, study_days):
    if not subjects.strip():
        return "Please enter your subjects."

    if not exam_dates.strip():
        return "Please enter your exam dates."

    messages = [
        SystemMessage(
            content=(
                "You are an AI Study Plan Generator. "
                "Create a realistic, organized study timetable. "
                "Prioritize subjects with earlier exams and difficult topics. "
                "Include breaks, revision, and practice questions. "
                "Do not schedule more study time than the user has available. "
                "If dates or details are unclear, state your assumptions. "
                "Use a clear table with day, time, subject, and activity."
            )
        ),
        HumanMessage(
            content=(
                f"Subjects: {subjects}\n"
                f"Available study hours per day: {hours}\n"
                f"Exam dates: {exam_dates}\n"
                f"Preferred study days: {study_days}\n\n"
                "Generate a practical study timetable leading up to the exams. "
                "Include daily tasks, short breaks, and revision sessions."
            )
        )
    ]

    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as error:
        return f"Error generating study plan: {error}"


# Create Gradio app
with gr.Blocks() as demo:
    gr.Markdown("# 📚 AI Study Plan Generator")
    gr.Markdown(
        "Enter your subjects, daily available hours, and exam dates "
        "to generate a personalized study timetable."
    )

    subjects = gr.Textbox(
        label="Subjects",
        placeholder="e.g. English, Maths, Physics, Computer Science",
        lines=3
    )

    hours = gr.Slider(
        minimum=1,
        maximum=12,
        value=3,
        step=1,
        label="Available Study Hours Per Day"
    )

    exam_dates = gr.Textbox(
        label="Exam Dates",
        placeholder="e.g. Maths: 10 November, English: 13 November",
        lines=3
    )

    study_days = gr.Dropdown(
        choices=[
            "Every day",
            "Monday to Friday",
            "Monday to Saturday",
            "Weekends only"
        ],
        value="Every day",
        label="Preferred Study Days"
    )

    generate_button = gr.Button(
        "Generate Study Plan",
        variant="primary"
    )

    output = gr.Markdown(label="Your Study Plan")

    generate_button.click(
        fn=generate_study_plan,
        inputs=[subjects, hours, exam_dates, study_days],
        outputs=output
    )

demo.launch()
