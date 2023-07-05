import os
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import messagebox

from data_handlers import DATASETS
from util import ACTIVATION_FUNCTIONS, get_logger


class Text:

    def __init__(self, root, text, row):
        tk.Label(root, text=text).grid(column=0, row=row, padx=10, pady=10, sticky=tk.W)

        self.widget = tk.Text(root, height=1, width=15)
        self.widget.grid(column=1, row=row)

    def get(self):
        return self.widget.get(1.0, "end-1c")


class OptionMenu:

    def __init__(self, root, text, row, values):
        tk.Label(root, text=text).grid(column=0, row=row, padx=10, pady=10, sticky=tk.W)
        
        self.var = tk.StringVar()
        self.var.set(values[0])
        tk.OptionMenu(root, self.var, *values).grid(column=1, row=row)

    def get(self):
        return self.var.get()


class LoggingWidget:

    def __init__(self, root, column, rowspan, path):
        self.widget = ScrolledText(root, state='disabled', wrap=tk.WORD)
        self.widget.grid(rowspan=rowspan, column=column, row=0, padx=10, pady=10)

        self.path = path

    def update(self, acc=-1):
        logs = []

        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                logs = "\n".join(f.readlines())

        self.widget.configure(state='normal')
        self.widget.delete('1.0', tk.END)
        self.widget.insert(tk.INSERT, logs)
        if acc > -1:
            self.widget.insert(tk.INSERT, f"\n\nAccuracy: {acc}\n\n")
        self.widget.configure(state='disabled')
        self.widget.yview(tk.END)

def h_layers_info():
    messagebox.showinfo(
        "Hidden layers",
        "Hidden layers are written as integers seperated by spaces. For example, the input '2 8 16' " +
        "will create a network with 3 hidden layers, the first has 2 nodes, the second has 8, and the " +
        "the third has 16."
    )

def gui(runner):
    root = tk.Tk()
    root.resizable(width=False, height=False)
    root.title('Task 3')

    dataset_inp = OptionMenu(root, "Select dataset: ", 0, DATASETS)

    hidden_layers_inp = Text(root, "Enter hidden layers: ", 1)
    mse_inp = Text(root, "Enter MSE threshold: ", 2)
    eta_inp = Text(root, "Enter learning rate: ", 3)
    epochs_inp = Text(root, "Enter number of epochs: ", 4)

    activation_inp = OptionMenu(root, "Select activation function: ", 5, list(ACTIVATION_FUNCTIONS.keys()))

    bias_var = tk.IntVar(root)
    tk.Checkbutton(root, text=" Bias", onvalue=1, offvalue=0, variable=bias_var).grid(column=0, row=6)

    logging_widget = LoggingWidget(root, 3, 7, "run.log")

    tk.Button(root, text="?", command=h_layers_info).grid(column=2, row=1)

    working = tk.Label(root, text="Running.....")


    def submit():
        h_layers = [int(num) for num in hidden_layers_inp.get().split()]
        mse = float(mse_inp.get())
        eta = float(eta_inp.get())
        epochs = int(epochs_inp.get())
        dataset = dataset_inp.get()
        activation = activation_inp.get()
        bias = int(bias_var.get())

        logger = get_logger(__name__)
        logger.info(f"GUI submitted for {dataset} with {h_layers} hidden layers, bias: {bias}, epochs: {epochs}, " +
                    f"activation: {activation}, learning rate: {eta}, MSE threshold: {mse}")

        working.grid(column=3, row=7, sticky=tk.W, padx=10, pady=10)
        root.update()

        acc = runner(h_layers, mse, eta, dataset, activation, bias, epochs)

        working.grid_forget()
        root.update()

        logging_widget.update(acc)


    tk.Button(root, text="Run", height=2, width=10, command=submit).grid(column=1, row=7, pady=10)

    logging_widget.update()
    root.mainloop()
