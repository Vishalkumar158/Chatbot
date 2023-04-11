import tkinter as tk
from tkinter import *
import training_duplicate as td



window = tk.Tk()

window.geometry("600x660")  # width x height of main window
window.configure(bg="#363636")
window.title("B.A.N.C. - Banking Automated Navigational ChatBot")

img = PhotoImage(file="chat_bot_64.jpg")
main_img = Label(window, image=img)
main_img.pack(pady=5)

# Creating space for displaying messages during the conversation.

main_frame = Frame(master=window, relief=tk.RAISED, borderwidth=5, bg="#363636")
scroll_bar = Scrollbar(main_frame)
message_space = Listbox(main_frame, width=90, height=20)
scroll_bar.pack(side=RIGHT, fill=Y, pady=60)
message_space.pack(side=LEFT, fill=BOTH, pady=60)
main_frame.pack(pady=0.5)

# Creating frames to display more text:
msg_on_top = tk.Label(text="B.A.N.C.", font=("Verdana", 20))
msg_on_top.place(x=240, y=95)

# Functions of buttons command :
def response_bot():

    query = input_text.get()
    message_space.insert(END, "You : " + query)
    ####################

    def final_response(x):
        td.speak(x)
        return x

    val = 0

    while True:
        texts_p = []

        # removing punctuation and converting to lowercase
        query = [letters.lower() for letters in str(query) if letters not in td.string.punctuation]
        query = ''.join(query)
        texts_p.append(query)
        # tokenizing and padding
        query = td.tokenizer.texts_to_sequences(texts_p)
        query = td.np.array(query).reshape(-1)
        query = td.pad_sequences([query], td.input_shape)
        # getting output from model
        output = td.model.predict(query)
        output = output.argmax()
        # finding the right tag and predicting
        response_tag = td.le.inverse_transform([output])[0]
        x = td.random.choice(td.responses[response_tag])
        if response_tag == "goodbye":
            ans = final_response(x)
            message_space.insert(END, "B.A.N.C. : " + ans)
            input_text.delete(0, END)
            val=1
            break
        else:
           break

    if val==0:
        answer = final_response(x)
        message_space.insert(END, "B.A.N.C. : " + answer)
        input_text.delete(0, END)

        # if response_tag == "goodbye":
        #     break

    ####################



# Creating input section

input_text_msg = tk.Label(text="Please enter your query here :", font=("Verdana", 20))
input_text = Entry(window, font=("Verdana", 20), bg="#363636", fg="white", borderwidth=5, border=5)
input_text_msg.place(x=90, y=480)
input_text.pack(fill=X, pady=5)

# Creating buttons for input :

button_1 = Button(window, text="Ask B.A.N.C.", font=("Verdana", 18), command=response_bot, relief=tk.RIDGE, border=3)
button_1.place(x=210, y=595)


window.mainloop()  # Creating a window for bot
