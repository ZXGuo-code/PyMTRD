import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import threading
import pymtrd_process as pp


def select_csv_file():
    """
    Select the configuration file
    the format of configuration file is csv
    """
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)


def run_program():
    """
    run program
    """
    configuration_file = file_entry.get()
    option = option_var.get()
    use_parallel = parallel_var.get()
    num_processes = process_entry.get() if use_parallel else '1'

    if not configuration_file:
        messagebox.showwarning("Input Error", "Please select a configuration file")
        return

    if use_parallel and not num_processes.isdigit():
        messagebox.showwarning("Input Error", "Please enter a valid number for thread count")
        return

    if (option == "Calculating the metrics of temporal rainfall distribution" or
            option == "Calculating the metrics of temporal rainfall distribution & Drawing"):
        bool_draw = 1
        if option == "Calculating the metrics of temporal rainfall distribution":
            bool_draw = 0
        if use_parallel:
            command = (f'python -c "import pymtrd_process as pp; pp.process_parallel(\'{configuration_file}\', '
                       f'num_processes={num_processes}, 'f'bool_draw={bool_draw})"')
        else:
            command = f'python -c "import pymtrd_process as pp; pp.process(\'{configuration_file}\', '\
                      f'bool_draw={bool_draw})"'
    else:
        command = f'python -c "import pymtrd_process as pp; pp.process_draw(\'{configuration_file}\')"'

    threading.Thread(target=execute_command, args=(command,)).start()


def execute_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in iter(process.stdout.readline, ''):
        output_text.insert(tk.END, line)
        output_text.see(tk.END)
    process.stdout.close()
    process.wait()
    for line in iter(process.stderr.readline, ''):
        output_text.insert(tk.END, line, 'error')
        output_text.see(tk.END)
    process.stderr.close()


def center_window(root):
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')


def toggle_process_entry():
    if parallel_var.get():
        process_entry.config(state='normal')
    else:
        process_entry.config(state='disabled')


def update_parallel_option(*args):
    if option_var.get() == "Drawing":
        parallel_check.config(state='disabled')
        process_entry.config(state='disabled')
        parallel_var.set(False)
    else:
        parallel_check.config(state='normal')


# create main window
root = tk.Tk()
root.title("PyMTRD")
# input configuration file
ttk.Label(root, text="Select Configuration File:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
file_entry = ttk.Entry(root, width=55)
file_entry.grid(row=0, column=1, padx=10, pady=10)
ttk.Button(root, text="Browse", command=select_csv_file).grid(row=0, column=2, padx=10, pady=10)
# data processing and drawing or drawing
ttk.Label(root, text="Select Processing Mode:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
option_var = tk.StringVar()
option_menu = ttk.Combobox(root, textvariable=option_var, values=[
    "Calculating the metrics of temporal rainfall distribution", "Drawing",
    "Calculating the metrics of temporal rainfall distribution & Drawing"], width=55)
option_menu.grid(row=1, column=1, padx=10, pady=10)
option_menu.current(0)
option_var.trace("w", update_parallel_option)
# use patallel computing or not
parallel_var = tk.BooleanVar()
parallel_check = ttk.Checkbutton(root, text="Use Parallel Computing", variable=parallel_var, command=toggle_process_entry)
parallel_check.grid(row=2, column=0, padx=10, pady=10, sticky="w")
# the num of threads used to calculate
ttk.Label(root, text="Processes Count:").grid(row=2, column=1, padx=10, pady=10, sticky="w")
process_entry = ttk.Entry(root, width=10)
process_entry.grid(row=2, column=1, padx=10, pady=10)
process_entry.config(state='disabled')
# run program button
ttk.Button(root, text="Run Program", command=run_program).grid(row=2, column=2, columnspan=3, pady=20)
# print context
output_text = tk.Text(root, wrap='word', height=15, width=90)
output_text.grid(row=4, column=0, columnspan=3, padx=10, pady=10)
output_text.tag_configure('error', foreground='red')

center_window(root)

root.mainloop()
