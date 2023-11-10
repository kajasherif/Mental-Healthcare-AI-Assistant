from langchain.agents import tool, Tool, load_tools
from langchain.schema import HumanMessage, SystemMessage
from utils import global_dict
import datetime

@tool
def schedule_therapy_session(message: str):
    """
    Use this tool to ask for day, time, and contact information from the user to schedule a therapy session.
    """
    # Get current date and time
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    
    if len(message) > 0:
        res = global_dict['llm_16k'](
            [SystemMessage(content="""These are required parameters to schedule a therapy session:

                            1. Date
                            2. Time
                            3. Email or phone Number
                                   
                            You create JSON only if you have the 3 parameters. If any parameter is missing you should ask the user to provide that."
                            Once you have the 3 parameters give the collected details in this format.
                            if you have email:
                            {
                            "date": "YYYY-MM-DD",
                            "time": "HH:MM:SS",
                            "userEmail": "user@example.com",
                            }
                            if you have phone number:
                            {
                            "date": "YYYY-MM-DD",
                            "time": "HH:MM:SS",
                            "userPhoneNumber": "1234567890",
                            }
                            if you have phone number and email:
                             {
                            "date": "YYYY-MM-DD",
                            "time": "HH:MM:SS",
                            "userEmail": "user@example.com",
                            "userPhoneNumber": "1234567890",
                            }
            """),
             HumanMessage(
                content=message
            )
            ])

        return res.content

    else:
        res = global_dict['llm'](
            [
                HumanMessage(
                    content=f"""Please ask the user to provide the day, time, and your contact information (email address or phone number) so we can schedule your therapy session. Keep your request polite and concise. Today's date is {current_date} and the current time is {current_time}."""
                )
            ]
        )

        return res.content


@tool
def mental_health_faq(question: str):
    """
    Use this tool to provide supportive responses to users sharing their mental health problems. Engage empathetically, ask for more details if needed, and offer solutions or suggest scheduling a therapy session in critical situations.
    """
    res = global_dict['llm'](
        [SystemMessage(content="""
                          When a user shares their mental health problems, respond supportively. Engage by asking for more details about what's bothering them. If the user explains their problem, comfort them with your response and provide solutions based on their issue. In cases of critical conditions, advise them to consult a therapist. If the user is open to it, use the schedule_therapy_session tool to arrange an appointment.But keep it short.
                          """),
            HumanMessage(
                content=f"""
                
                Question: {question}"""
        )
        ]
    )

    return res.content


tools_diff = [
    Tool(
        name="mental_health_faq",
        func=mental_health_faq,
        description="""Use this tool to empathetically engage with users facing mental health issues. Provide supportive responses, ask for more details, and offer solutions. In critical situations, suggest consulting a therapist and use the schedule_therapy_session tool if the user agrees.""",
        return_direct=True
    ),
    Tool(
        name="schedule_therapy_session",
        func=schedule_therapy_session,
        description="""
Use this tool to schedule a therapy session after collecting the date, time, and contact information from the user.
""",
        return_direct=True
    ),
]
