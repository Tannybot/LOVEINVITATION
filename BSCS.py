import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk

# Define messages in order (If ever di mo yes si gorl)
messages = [
    "Hewo my sugarplum sweetiecakes, can I ask you for a date?",
    "You dont want me anymore loveyboo?",
    "Just say yes, for me to dress up as your loverboy",
    "No?? O-okay"
]

def ask_out(index=0):
    hesitant_message = messages[index]
    answer = messagebox.askyesno("Date Invitation", hesitant_message)
    if answer:
        messagebox.showinfo("Response", "Great! Let's plan something my sugarplum.")
        select_date_option()
    elif index < len(messages) - 1:
        ask_out(index + 1)
    else:
        messagebox.showinfo("Response", "That's alright. Maybe another time.")
        # Stop iteration
        return

def select_date_option():
    options = ["Dinner", "Movie", "Coffee", "Walk in the Park", "Concert", "Special Venue???"]
    selected_option = tk.StringVar(root)
    selected_option.set(options[0])

    label.config(text="Select a date option:")
    button.config(text="Confirm")

    dropdown = ttk.Combobox(root, textvariable=selected_option, values=options)
    dropdown.pack(pady=240)

    def confirm_option():
        selected = selected_option.get()
        messagebox.showinfo("Date Option", "You've chosen: {selected}. Enjoy your date and make something happen!! O.O")

    button.config(command=confirm_option)

# main application window (Title sa window )
root = tk.Tk()
root.title("Date Invitation")

# Change bg for window
root.configure(bg='#FFD700')
root.geometry("1000x500")

# Allow resizing of the window
root.resizable(True, True)

# Create frame for design
design_frame = tk.Frame(root, bg="#FF69B4", bd=5)
design_frame.place(relx=0.5, rely=0.5, relwidth=0.9, relheight=0.9, anchor='center')

# Load the image and resize it
image = Image.open("C:\\Users\\creen\\Downloads\\76a.PNG")
image = image.resize((400, 300), Image.Resampling.LANCZOS)
photo = ImageTk.PhotoImage(image)

# Create a label to display the image inside the frame
image_label = tk.Label(design_frame, image=photo)
image_label.pack()


# Label for invitation message
label = tk.Label(design_frame, text="Hey there my Sugarcakes, press the button below, I've got something for ya", bg='#FFD700', font=("Helvetica", 14, "bold"))
label.pack(padx=20, pady=20)

# Button to ask for the date
button = tk.Button(design_frame, text="Start the Love!", command=ask_out, bg='black', fg='white', font=("Helvetica", 12, "bold"))
button.pack(pady=10)

root.mainloop()
