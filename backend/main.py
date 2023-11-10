from agent import initialize_sales_agent
from utils import (
    intialize_model_and_vector_store,
    #run_hybrid_agent,
    global_dict,
    #get_chat_history_gradio,
    conversation_stages,
    check_intermediate_steps,
    check_relevance,
    generate_with_llm
)
import gradio as gr
from gradio.themes.utils.colors import Color
from langchain.callbacks import get_openai_callback

text_color = "#FFFFFF"
app_background = "#0A0A0A"
user_inputs_background = "#193C4C"#14303D"#"#091820"
widget_bg = "#000100"
button_bg = "#141414"

#https://discuss.huggingface.co/t/is-there-a-way-to-force-the-dark-mode-theme-in-gradio/22314/2

dark = Color(
    name="dark",
    c50="#F4F3EE",  # not sure
    # all text color:
    c100=text_color, # Title color, input text color, and all chat text color.
    c200=text_color, # Widget name colors (system prompt and "chatbot")
    c300="#F4F3EE", # not sure
    c400="#F4F3EE", # Possibly gradio link color. Maybe other unlicked link colors.
    # suggestion text color...
    c500=text_color, # text suggestion text. Maybe other stuff.
    c600=button_bg,#"#444444", # button background color, also outline of user msg.
    # user msg/inputs color:
    c700=user_inputs_background, # text input background AND user message color. And bot reply outline.
    # widget bg.
    c800=widget_bg, # widget background (like, block background. Not whole bg), and bot-reply background.
    c900=app_background, # app/jpage background. (v light blue)
    c950="#F4F3EE", # not sure atm. 
)




if __name__ == "__main__":
    intialize_model_and_vector_store()
    initialize_sales_agent()
    #run_hybrid_agent()
    
    
    
    def chat(user,chat_history=global_dict["actual_memory"]):
        
        global_dict["actual_memory"].append(f"User: {user}")
        chat_hist_1="\n".join(global_dict["actual_memory"][-2:])
        
        with get_openai_callback() as cb:
        
            validation_output=global_dict["validation_chain"].run(chat_history=chat_hist_1)
            print(user)
            print(validation_output)
            #check if comes in category 1 or 2
            if (validation_output in ["1","1.","1. Relevant"]) or (validation_output !="3"):
                
                ## if it is 2, assign validation output as user input
                if validation_output not in ["1","1.","1. Relevant"]:
                    user=generate_with_llm(chat_hist_1)
                    print(f"New:{user}")
            
                #global_dict["actual_memory"].append(f"User: {user}")
                
                chat_hist="\n".join(global_dict["actual_memory"])
                # relevance_score=check_relevance(user)
                # if relevance_score>=0.54:
                #     return "I'm sorry,I can only help in answering questions about iNextLabs and it solutions."
                # else:    
                
                
                output=global_dict["agent"](user)
                
                #chat_hist=get_chat_history_gradio(global_dict['agent_memory'])
                

                
                #chat_hist=chat_history
                # print("chat_history:")
                # print(chat_hist)
                context=output["output"]
                
                
                conv_stage_no=global_dict['stage_analyzer_chain'].run(conversation_history=chat_hist)

                conv_stage=conversation_stages.get(conv_stage_no,"1")
                
                check_intermediate_steps(output)
                    
                    
                if global_dict['prod_questions']>=5:
                    conv_stage=conversation_stages.get("3")
                
                if context=="Sorry, no relevant information is available to answer your query.":
                    context="I don't have information to answer your query"    
                    
                if len(chat_hist)==0:
                    final_utterance=global_dict['conversation_utterance_chain'].run(conversation_stage=conv_stage,
                                                        conversation_history="",
                                                        context="")
                    
                else:
                    print(global_dict['prod_questions'])
                    print(f"CONV STAGE: {conv_stage}")
                    final_utterance=global_dict['conversation_utterance_chain'].run(conversation_stage=conv_stage,
                                                        conversation_history=chat_hist_1,
                                                        context=context) 
                    
                
                
                global_dict["actual_memory"].append(f"Aibo: {final_utterance}")
                
                print(f'Spent a total of {cb.total_tokens} tokens')
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Cost (USD): ${cb.total_cost}")
                
                return final_utterance
            else:
                response="I can help with information about iNextLabs."
                
                return response
        
    with gr.Blocks(theme=gr.themes.Monochrome(
        font=[gr.themes.GoogleFont("Montserrat"), "Arial", "sans-serif"],
        primary_hue="sky",  # when loading
        secondary_hue="sky", # something with links
        neutral_hue="dark"),) as demo:  #main.
        
        chatbot = gr.ChatInterface(fn=chat)
    
    demo.launch()    