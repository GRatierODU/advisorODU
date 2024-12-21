from advisor import build_workflow, AgentState
from threading import Thread
from tkinter import ttk
import tkinter as tk
import logging
import json
import os


# ----------------------------------
# Logging Configuration
# ----------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s:  %(message)s"
)
logger = logging.getLogger(__name__)


# ----------------------------------
# Loading Animation
# ----------------------------------
class LoadingSpinner:
    def __init__(self, parent):
        self.parent = parent
        self.label = tk.Label(
            parent, text="", font=("Courier New", 24), fg="black"
        )  # Larger font and black text
        self.frames = ["|", "/", "-", "\\"]
        self.current_frame = 0
        self.running = False

    def start(self):
        self.running = True
        self.label.place(
            relx=0.05, rely=0.88, anchor="center"
        )  # Centered between input and border
        self.animate()

    def stop(self):
        self.running = False
        self.label.place_forget()

    def animate(self):
        if self.running:
            frame = self.frames[self.current_frame]
            self.label.config(text=frame)
            self.current_frame = (self.current_frame + 1) % len(self.frames)
            self.parent.after(100, self.animate)  # Update every 100ms


# ----------------------------------
# Build Chatbot UI App
# ----------------------------------
class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ODU Academic Advisor AI")

        # Load avatars
        self.user_avatar = tk.PhotoImage(file="./images/user_avatar.png")
        self.ai_avatar = tk.PhotoImage(file="./images/ai_avatar.png")

        # Enable resizing
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=0)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)

        # Chat display frame
        self.chat_display_frame = ttk.Frame(root)
        self.chat_display_frame.grid(
            row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew"
        )

        # Chat display text area
        self.chat_display = tk.Text(
            self.chat_display_frame,
            wrap="word",
            state="disabled",
            font=("Courier New", 12, "italic"),
            bg="#1e1e2e",  # Original dark background
            fg="#cdd6f4",  # Original light text
            insertbackground="#f38ba8",  # Original accent color (caret)
            relief=tk.FLAT,
            padx=10,
            pady=10,
        )

        self.chat_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar for the chat display
        self.scrollbar = ttk.Scrollbar(
            self.chat_display_frame, command=self.chat_display.yview
        )
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.chat_display.config(yscrollcommand=self.scrollbar.set)

        # Input frame for spinner and input text
        input_frame = ttk.Frame(root)
        input_frame.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        # Spinner for AI processing
        self.spinner = LoadingSpinner(self.root)

        # User input field
        self.user_input = tk.Entry(root, width=70, font=("Arial", 12))
        self.user_input.grid(row=1, column=0, padx=10, pady=10)
        self.user_input.bind("<Return>", self.handle_user_input)

        # Send button
        self.send_button = tk.Button(
            root, text="Send", command=self.handle_user_input, font=("Arial", 12)
        )
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        # Feedback frame
        feedback_frame = tk.Frame(root)  # Frame to group feedback components
        feedback_frame.grid(row=2, column=0, columnspan=2, pady=(5, 10))

        # Label to display slider value
        self.slider_value_label = tk.Label(
            feedback_frame, text="Feedback Score: 3", font=("Arial", 12)
        )
        self.slider_value_label.pack(side="left", padx=(10, 10))

        # Slider for feedback (1-5)
        self.feedback_slider = ttk.Scale(
            feedback_frame,
            from_=1,
            to=5,
            orient="horizontal",
            length=300,
            command=self.update_slider_label,  # Call method whenever slider value changes
        )
        self.feedback_slider.set(3)  # Default position
        self.feedback_slider.pack(side="left", padx=(10, 10))

        # Submit Feedback button
        self.submit_feedback_button = tk.Button(
            feedback_frame,
            text="Submit Feedback",
            command=self.open_feedback_dialog,
            state="disabled",  # Initially disabled
            font=("Arial", 12),
        )
        self.submit_feedback_button.pack(side="left", padx=(10, 10))

        # Initialize chatbot
        self.memory = []
        self.workflow = build_workflow()

        # Draw workflow
        self.workflow.get_graph().draw_mermaid_png(
            output_file_path="./images/graph.png"
        )

        self.current_ai_response = None  # Keep track of the current AI response

        # Greeting
        self.add_chat_message(
            "AI", "Welcome to ODU Academic Advisor AI! Type your question below."
        )

    # Method to update slider value label
    def update_slider_label(self, value):
        self.slider_value_label.config(text=int(float(value)))

    # Method to enable/disable input and start/stop spinnerdef toggle_input(self, enable):
    def toggle_input(self, enable):
        if enable:
            self.user_input.config(state="normal")
            self.user_input.delete(0, tk.END)  # Clear "Thinking..."
            self.send_button.config(state="normal")
            self.spinner.stop()  # Stop the spinner animation
        else:
            self.user_input.config(state="normal")  # Temporarily enable to insert text
            self.user_input.delete(0, tk.END)
            self.user_input.insert(0, "Thinking...")  # Insert "Thinking..."
            self.user_input.config(state="disabled")  # Disable again
            self.send_button.config(state="disabled")
            self.spinner.start()  # Start the spinner animation

    # Add message to chat display
    def add_chat_message(self, sender, message):
        """Adds a chat message to the display with avatars."""
        self.chat_display.configure(state="normal")
        tag = "user" if sender == "You" else "ai"
        avatar = self.user_avatar if sender == "You" else self.ai_avatar

        # Insert avatar and message
        self.chat_display.image_create(tk.END, image=avatar)
        self.chat_display.insert(tk.END, f" {sender}: {message}\n\n", tag)

        # Configure message styling
        self.chat_display.tag_config(
            "user", foreground="#89ddff", font=("Courier New", 12)
        )
        self.chat_display.tag_config(
            "ai", foreground="#cdd6f4", font=("Courier New", 12)
        )

        self.chat_display.configure(state="disabled")
        self.chat_display.see(tk.END)

    # Handles user input from the entry field or Send button
    def handle_user_input(self, event=None):
        """Handles user input and starts processing."""
        user_message = self.user_input.get().strip()
        if user_message:
            self.add_chat_message("You", user_message)
            self.user_input.delete(0, tk.END)

            self.add_chat_message("AI", "Typing...")
            self.toggle_input(False)  # Disable input and start spinner
            Thread(target=self.get_ai_response, args=(user_message,)).start()

    # Process the user message and fetch AI response
    def get_ai_response(self, user_message):
        try:
            self.chat_display.configure(state="normal")
            self.chat_display.delete("end-3l", "end-2l")  # Remove "Typing..." message
            self.chat_display.configure(state="disabled")

            # Disable feedback buttons during processing
            self.submit_feedback_button.config(state="disabled")
            self.memory.append({"type": "human", "content": user_message})
            state = AgentState(
                user_input=user_message,
                memory=self.memory,
            )
            output = self.workflow.invoke(state)
            self.memory.append({"type": "ai", "content": output["revised_answer"]})
            self.current_ai_response = output["revised_answer"]
            self.stream_response("AI", output["revised_answer"])

        except Exception:
            logging.exception("Error fetching AI response")
            self.add_chat_message(
                "AI", "Sorry, something went wrong. Please try again."
            )
        finally:
            self.toggle_input(True)  # Disable input and start spinner
            self.submit_feedback_button.config(state="normal")

    def stream_response(self, sender, response, delay=100):
        """Streams the response chunk by chunk, including avatar and prefix."""
        # Insert the avatar and prefix first
        self.chat_display.configure(state="normal")
        self.chat_display.image_create(tk.END, image=self.ai_avatar)  # Add avatar
        self.chat_display.insert(
            tk.END, f" {sender}:\n", "ai"
        )  # Add prefix and newline
        self.chat_display.configure(state="disabled")
        self.chat_display.see(tk.END)

        # Split the response into words or sentences for streaming
        chunks = response.split(" ")  # Word-by-word streaming

        def stream_chunk(i=0):
            if i < len(chunks):
                # Append the next chunk
                self.chat_display.configure(state="normal")
                self.chat_display.insert(tk.END, f"{chunks[i]} ", "ai")
                self.chat_display.configure(state="disabled")
                self.chat_display.see(tk.END)

                # Schedule the next chunk
                self.root.after(delay, stream_chunk, i + 1)
            else:
                # Ensure proper spacing after completing streaming
                self.chat_display.configure(state="normal")
                self.chat_display.insert(tk.END, "\n\n", "ai")
                self.chat_display.configure(state="disabled")
                self.submit_feedback_button.config(state="normal")

        # Start streaming
        stream_chunk()

    def open_feedback_dialog(self):
        # Open a new window for feedback reason
        dialog = tk.Toplevel(self.root)
        dialog.title("Feedback Reason")
        dialog.geometry("400x200")

        label = tk.Label(
            dialog,
            text="Please provide a reason for your feedback (optional):",
            font=("Arial", 12),
        )
        label.pack(pady=10)

        text_box = tk.Text(dialog, height=5, width=40, font=("Arial", 12))
        text_box.pack(pady=10)

        submit_button = tk.Button(
            dialog,
            text="Submit",
            command=lambda: self.submit_feedback(
                dialog, text_box.get("1.0", tk.END).strip()
            ),
            font=("Arial", 12),
        )
        submit_button.pack(pady=10)

    # Handle feedback buttons from the user
    def submit_feedback(self, dialog, reason):
        dialog.destroy()

        feedback_score = int(self.feedback_slider.get())

        # Submit feedback to the learning mechanism
        feedback_data = {
            "question": self.memory[-2]["content"],
            "answer": self.current_ai_response,
            "feedback": feedback_score,
            "reason": reason,
        }

        # Save feedback data to a JSON file
        self.save_feedback(feedback_data)
        self.add_chat_message(
            "AI",
            f"Thank you for your feedback (Score: {feedback_score}, Reason: {reason})!",
        )

        # Disable the feedback button after submission
        self.submit_feedback_button.config(state="disabled")

    # Save both good and bad feedback
    def save_feedback(self, feedback_data):
        """Save feedback to a JSON file."""
        feedback_file = "./data/feedback_data.json"

        # Ensure the file exists and is properly initialized
        if not os.path.exists(feedback_file) or os.path.getsize(feedback_file) == 0:
            # Create the file and initialize it with an empty list
            with open(feedback_file, "w") as f:
                json.dump([], f)

        # Load existing data safely
        try:
            with open(feedback_file, "r") as f:
                feedback_list = json.load(f)  # Load existing JSON data
        except json.JSONDecodeError:
            logger.warning(f"{feedback_file} is invalid. Reinitializing as empty.")
            feedback_list = []  # Reset to empty if the file is malformed

        # Append new feedback
        feedback_list.append(feedback_data)

        # Save updated data
        with open(feedback_file, "w") as f:
            json.dump(feedback_list, f, indent=4)


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()
