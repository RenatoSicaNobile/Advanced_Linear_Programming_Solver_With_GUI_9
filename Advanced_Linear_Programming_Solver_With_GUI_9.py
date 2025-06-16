import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Polygon, FancyArrowPatch
import json
import os
from matplotlib.ticker import MaxNLocator

class LinearProgrammingSolver:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced LP/ILP Solver - Università")
        self.root.geometry("1400x950")
        
        # Initialize variables
        self.variables = ["x₁", "x₂"]
        self.constraints = []
        self.solution = None
        self.slack_values = None
        self.optimal_value = None
        self.dual_solution = None
        self.alternative_solutions = []
        self.integer_vars = []
        self.problem_history = []
        
        # Initialize problem parameters
        self.obj_type = tk.StringVar(value="max")
        self.obj_coeffs = []
        self.obj_signs = []
        
        # Create GUI
        self.create_widgets()
        self.set_default_problem()
        
    def create_widgets(self):
        # Main frames
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = tk.Frame(main_frame, width=500)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel widgets
        self.create_objective_function_section(left_frame)
        self.create_constraints_section(left_frame)
        self.create_non_negativity_section(left_frame)
        self.create_integer_vars_section(left_frame)
        self.create_control_buttons(left_frame)
        
        # Right panel widgets
        self.create_output_section(right_frame)
        self.create_graph_section(right_frame)
        self.create_solution_steps_section(right_frame)
    
    def create_objective_function_section(self, parent):
        frame = tk.LabelFrame(parent, text="Funzione Obiettivo")
        frame.pack(fill=tk.X, pady=5)
        
        # Optimization type (min/max)
        ttk.Combobox(frame, textvariable=self.obj_type, values=["min", "max"], width=5, state="readonly").pack(side=tk.LEFT)
        tk.Label(frame, text=" z = ").pack(side=tk.LEFT)
        
        # Clear previous entries
        self.obj_coeffs.clear()
        self.obj_signs.clear()
        
        # Create entries for each variable
        for var in self.variables:
            sign = ttk.Combobox(frame, values=["+", "-"], width=2, state="readonly")
            sign.set("+")
            sign.pack(side=tk.LEFT)
            self.obj_signs.append(sign)
            
            coeff = tk.Entry(frame, width=5)
            coeff.insert(0, "0")
            coeff.pack(side=tk.LEFT)
            self.obj_coeffs.append(coeff)
            
            tk.Label(frame, text=var).pack(side=tk.LEFT)
    
    def create_constraints_section(self, parent):
        frame = tk.LabelFrame(parent, text="Vincoli")
        frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create scrollable canvas for constraints
        canvas = tk.Canvas(frame)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        self.constraints_container = tk.Frame(canvas)
        
        self.constraints_container.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.constraints_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_non_negativity_section(self, parent):
        frame = tk.LabelFrame(parent, text="Non Negatività")
        frame.pack(fill=tk.X, pady=5)
        
        self.nn_signs = []
        self.nn_values = []
        
        for var in self.variables:
            inner = tk.Frame(frame)
            inner.pack(anchor="w")
            
            tk.Label(inner, text=var).pack(side=tk.LEFT)
            
            sign = ttk.Combobox(inner, values=["≥", "≤", "Libera"], width=6, state="readonly")
            sign.set("≥")
            sign.pack(side=tk.LEFT)
            self.nn_signs.append(sign)
            
            val = tk.Entry(inner, width=5)
            val.insert(0, "0")
            val.pack(side=tk.LEFT)
            self.nn_values.append(val)
    
    def create_integer_vars_section(self, parent):
        frame = tk.LabelFrame(parent, text="Variabili Intere")
        frame.pack(fill=tk.X, pady=5)
        
        self.integer_vars = []
        
        for var in self.variables:
            var_frame = tk.Frame(frame)
            var_frame.pack(anchor="w")
            
            chk_var = tk.IntVar()
            chk = tk.Checkbutton(var_frame, text=var, variable=chk_var)
            chk.pack(side=tk.LEFT)
            self.integer_vars.append(chk_var)
    
    def create_control_buttons(self, parent):
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, pady=10)
        
        # Main action buttons
        tk.Button(frame, text="RISOLVI", command=self.solve_problem, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=2)
        tk.Button(frame, text="RESET", command=self.reset_problem, bg="#f44336", fg="white").pack(side=tk.LEFT, padx=2)
        tk.Button(frame, text="ESPORTA PDF", command=self.export_pdf, bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=2)
        
        # Variable management
        tk.Button(frame, text="+ VARIABILE", command=self.add_variable).pack(side=tk.LEFT, padx=2)
        tk.Button(frame, text="- VARIABILE", command=self.remove_variable).pack(side=tk.LEFT, padx=2)
        
        # Constraint management
        tk.Button(frame, text="+ VINCOLO", command=self.add_constraint).pack(side=tk.LEFT, padx=2)
        
        # Problem management
        tk.Button(frame, text="SALVA PROBLEMA", command=self.save_problem, bg="#9C27B0", fg="white").pack(side=tk.LEFT, padx=2)
        tk.Button(frame, text="CARICA PROBLEMA", command=self.load_problem, bg="#FF9800", fg="white").pack(side=tk.LEFT, padx=2)
        
        # Graph options
        tk.Button(frame, text="GRAFICO", command=self.show_graph_window, bg="#607D8B", fg="white").pack(side=tk.LEFT, padx=2)
    
    def create_output_section(self, parent):
        frame = tk.LabelFrame(parent, text="Soluzione")
        frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.output_text = tk.Text(frame, wrap=tk.WORD, height=10, font=("Courier New", 10))
        scrollbar = tk.Scrollbar(frame, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.output_text.config(yscrollcommand=scrollbar.set)
    
    def create_solution_steps_section(self, parent):
        frame = tk.LabelFrame(parent, text="Passaggi della Soluzione")
        frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.steps_text = tk.Text(frame, wrap=tk.WORD, height=15, font=("Courier New", 9))
        scrollbar = tk.Scrollbar(frame, command=self.steps_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.steps_text.pack(fill=tk.BOTH, expand=True)
        self.steps_text.config(yscrollcommand=scrollbar.set)
    
    def create_graph_section(self, parent):
        frame = tk.LabelFrame(parent, text="Regione Ammissibile")
        frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.figure = plt.Figure(figsize=(6, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, frame)
        self.toolbar.update()
    
    def set_default_problem(self):
        """Set up the problem with all zeros"""
        self.obj_type.set("max")
        
        # Set objective function coefficients to zero
        for i in range(len(self.variables)):
            self.obj_signs[i].set("+")
            self.obj_coeffs[i].delete(0, tk.END)
            self.obj_coeffs[i].insert(0, "0")
        
        # Clear all constraints first
        for constraint in self.constraints:
            constraint[0].destroy()
        self.constraints = []
        
        # Add one empty constraint
        self.add_constraint()
        
        # Set non-negativity constraints
        for i in range(len(self.variables)):
            self.nn_signs[i].set("≥")
            self.nn_values[i].delete(0, tk.END)
            self.nn_values[i].insert(0, "0")
            self.integer_vars[i].set(0)
    
    def add_variable(self):
        """Add a new variable to the problem"""
        next_index = len(self.variables) + 1
        self.variables.append(f"x_{next_index}")
        
        # Rebuild the interface to reflect the new variable
        self.rebuild_interface()
    
    def remove_variable(self):
        """Remove the last variable from the problem"""
        if len(self.variables) > 2:
            self.variables.pop()
            self.rebuild_interface()
    
    def rebuild_interface(self):
        """Rebuild the entire interface to reflect variable changes"""
        # Save current values
        saved_obj = [(sign.get(), coeff.get()) for sign, coeff in zip(self.obj_signs, self.obj_coeffs)]
        saved_constraints = []
        
        for constraint in self.constraints:
            signs = [s.get() for s in constraint[1]]
            coeffs = [e.get() for e in constraint[2]]
            ineq = constraint[3].get()
            rhs = constraint[4].get()
            saved_constraints.append((signs, coeffs, ineq, rhs))
        
        saved_nn = [(sign.get(), val.get()) for sign, val in zip(self.nn_signs, self.nn_values)]
        saved_int = [var.get() for var in self.integer_vars]
        
        # Destroy all widgets and reinitialize
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Reinitialize with current variables
        self.__init__(self.root)
        
        # Restore objective function
        for i, (sign, coeff) in enumerate(saved_obj):
            if i < len(self.obj_signs):
                self.obj_signs[i].set(sign)
                self.obj_coeffs[i].delete(0, tk.END)
                self.obj_coeffs[i].insert(0, coeff)
        
        # Restore constraints
        for signs, coeffs, ineq, rhs in saved_constraints:
            # Pad coefficients if we added variables
            while len(coeffs) < len(self.variables):
                coeffs.append("0")
                signs.append("+")
            
            # Truncate if we removed variables
            if len(coeffs) > len(self.variables):
                coeffs = coeffs[:len(self.variables)]
                signs = signs[:len(self.variables)]
            
            # Convert to numerical values
            try:
                coeff_values = [float(c) if s == "+" else -float(c) for c, s in zip(coeffs, signs)]
                rhs_value = float(rhs)
                self.add_constraint_with_values(coeff_values, ineq, rhs_value)
            except ValueError:
                pass
        
        # Restore non-negativity constraints
        for i, (sign, val) in enumerate(saved_nn):
            if i < len(self.nn_signs):
                self.nn_signs[i].set(sign)
                self.nn_values[i].delete(0, tk.END)
                self.nn_values[i].insert(0, val)
        
        # Restore integer variables
        for i, val in enumerate(saved_int):
            if i < len(self.integer_vars):
                self.integer_vars[i].set(val)
    
    def add_constraint(self):
        """Add a new empty constraint"""
        frame = tk.Frame(self.constraints_container)
        frame.pack(anchor="w", pady=2)
        
        entries = []
        signs = []
        
        for var in self.variables:
            sign = ttk.Combobox(frame, values=["+", "-"], width=2, state="readonly")
            sign.set("+")
            sign.pack(side=tk.LEFT)
            signs.append(sign)
            
            ent = tk.Entry(frame, width=5)
            ent.insert(0, "0")
            ent.pack(side=tk.LEFT)
            entries.append(ent)
            
            tk.Label(frame, text=var).pack(side=tk.LEFT)
        
        ineq = ttk.Combobox(frame, values=["≤", "≥", "="], width=3, state="readonly")
        ineq.set("≤")
        ineq.pack(side=tk.LEFT)
        
        rhs = tk.Entry(frame, width=5)
        rhs.insert(0, "0")
        rhs.pack(side=tk.LEFT)
        
        # Add delete button for this constraint
        del_btn = tk.Button(frame, text="✕", command=lambda: self.remove_constraint(frame), 
                          fg="red", font=("Arial", 8), bd=0)
        del_btn.pack(side=tk.LEFT, padx=5)
        
        self.constraints.append((frame, signs, entries, ineq, rhs))
    
    def add_constraint_with_values(self, coeffs, inequality, rhs):
        """Helper method to add constraints with predefined values"""
        frame = tk.Frame(self.constraints_container)
        frame.pack(anchor="w", pady=2)
        
        entries = []
        signs = []
        
        for i, coeff in enumerate(coeffs):
            sign = ttk.Combobox(frame, values=["+", "-"], width=2, state="readonly")
            sign.set("+" if coeff >= 0 else "-")
            sign.pack(side=tk.LEFT)
            signs.append(sign)
            
            ent = tk.Entry(frame, width=5)
            ent.insert(0, str(abs(coeff)))
            ent.pack(side=tk.LEFT)
            entries.append(ent)
            
            tk.Label(frame, text=self.variables[i]).pack(side=tk.LEFT)
        
        ineq = ttk.Combobox(frame, values=["≤", "≥", "="], width=3, state="readonly")
        ineq.set(inequality)
        ineq.pack(side=tk.LEFT)
        
        rhs_entry = tk.Entry(frame, width=5)
        rhs_entry.insert(0, str(rhs))
        rhs_entry.pack(side=tk.LEFT)
        
        # Add delete button for this constraint
        del_btn = tk.Button(frame, text="✕", command=lambda: self.remove_constraint(frame), 
                          fg="red", font=("Arial", 8), bd=0)
        del_btn.pack(side=tk.LEFT, padx=5)
        
        self.constraints.append((frame, signs, entries, ineq, rhs_entry))
    
    def remove_constraint(self, frame):
        """Remove a specific constraint"""
        for i, constraint in enumerate(self.constraints):
            if constraint[0] == frame:
                frame.destroy()
                self.constraints.pop(i)
                break
    
    def reset_problem(self):
        """Reset the problem to default state with all zeros"""
        # Clear all variables and constraints
        self.variables = ["x₁", "x₂"]
        
        # Rebuild interface with default values
        self.rebuild_interface()
        
        # Set all entries to zero
        for coeff in self.obj_coeffs:
            coeff.delete(0, tk.END)
            coeff.insert(0, "0")
        
        for sign in self.obj_signs:
            sign.set("+")
        
        for sign in self.nn_signs:
            sign.set("≥")
        
        for val in self.nn_values:
            val.delete(0, tk.END)
            val.insert(0, "0")
        
        for var in self.integer_vars:
            var.set(0)
        
        # Clear all constraints
        for constraint in self.constraints:
            constraint[0].destroy()
        self.constraints = []
        
        # Add one empty constraint
        self.add_constraint()
    
    def save_problem(self):
        """Save the current problem to a file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Salva problema"
        )
        
        if not filename:
            return
        
        try:
            problem_data = {
                "variables": self.variables,
                "obj_type": self.obj_type.get(),
                "obj_coeffs": [entry.get() for entry in self.obj_coeffs],
                "obj_signs": [sign.get() for sign in self.obj_signs],
                "constraints": [
                    {
                        "signs": [sign.get() for sign in constraint[1]],
                        "coeffs": [entry.get() for entry in constraint[2]],
                        "ineq": constraint[3].get(),
                        "rhs": constraint[4].get()
                    }
                    for constraint in self.constraints
                ],
                "nn_signs": [sign.get() for sign in self.nn_signs],
                "nn_values": [val.get() for val in self.nn_values],
                "integer_vars": [var.get() for var in self.integer_vars]
            }
            
            with open(filename, "w") as f:
                json.dump(problem_data, f, indent=4)
            
            messagebox.showinfo("Successo", f"Problema salvato con successo:\n{filename}")
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile salvare il problema:\n{str(e)}")
    
    def load_problem(self):
        """Load a problem from a file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON Files", "*.json")],
            title="Carica problema"
        )
        
        if not filename:
            return
        
        try:
            with open(filename, "r") as f:
                problem_data = json.load(f)
            
            # Set variables
            self.variables = problem_data["variables"]
            
            # Rebuild interface
            self.rebuild_interface()
            
            # Set objective function
            self.obj_type.set(problem_data["obj_type"])
            for i, (sign, coeff) in enumerate(zip(problem_data["obj_signs"], problem_data["obj_coeffs"])):
                if i < len(self.obj_signs):
                    self.obj_signs[i].set(sign)
                    self.obj_coeffs[i].delete(0, tk.END)
                    self.obj_coeffs[i].insert(0, coeff)
            
            # Clear existing constraints
            for constraint in self.constraints:
                constraint[0].destroy()
            self.constraints = []
            
            # Add constraints from file
            for constraint in problem_data["constraints"]:
                signs = constraint["signs"]
                coeffs = constraint["coeffs"]
                ineq = constraint["ineq"]
                rhs = constraint["rhs"]
                
                # Pad coefficients if needed
                while len(coeffs) < len(self.variables):
                    coeffs.append("0")
                    signs.append("+")
                
                # Convert to numerical values
                try:
                    coeff_values = [float(c) if s == "+" else -float(c) for c, s in zip(coeffs, signs)]
                    rhs_value = float(rhs)
                    self.add_constraint_with_values(coeff_values, ineq, rhs_value)
                except ValueError:
                    pass
            
            # Set non-negativity constraints
            for i, (sign, val) in enumerate(zip(problem_data["nn_signs"], problem_data["nn_values"])):
                if i < len(self.nn_signs):
                    self.nn_signs[i].set(sign)
                    self.nn_values[i].delete(0, tk.END)
                    self.nn_values[i].insert(0, val)
            
            # Set integer variables
            for i, val in enumerate(problem_data["integer_vars"]):
                if i < len(self.integer_vars):
                    self.integer_vars[i].set(val)
            
            messagebox.showinfo("Successo", f"Problema caricato con successo:\n{filename}")
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile caricare il problema:\n{str(e)}")
    
    def solve_problem(self):
        """Solve the linear programming problem"""
        try:
            # Clear previous results
            self.output_text.delete(1.0, tk.END)
            self.steps_text.delete(1.0, tk.END)
            self.ax.clear()
            self.problem_history = []
            
            # Record problem statement
            self.steps_text.insert(tk.END, "=== PROBLEMA INIZIALE ===\n\n")
            
            # Objective function
            obj_str = f"{self.obj_type.get()} z = "
            for i, (sign, entry) in enumerate(zip(self.obj_signs, self.obj_coeffs)):
                coeff = float(entry.get())
                coeff = coeff if sign.get() == "+" else -coeff
                if i > 0 and coeff >= 0:
                    obj_str += "+ "
                obj_str += f"{coeff} {self.variables[i]} "
            self.steps_text.insert(tk.END, obj_str + "\n\n")
            
            # Constraints
            self.steps_text.insert(tk.END, "Vincoli:\n")
            for i, (_, signs, entries, ineq, rhs) in enumerate(self.constraints):
                constr_str = ""
                for j, (sign, entry) in enumerate(zip(signs, entries)):
                    val = float(entry.get())
                    val = val if sign.get() == "+" else -val
                    if j > 0 and val >= 0:
                        constr_str += "+ "
                    constr_str += f"{val} {self.variables[j]} "
                constr_str += f"{ineq.get()} {rhs.get()}"
                self.steps_text.insert(tk.END, f"  {constr_str}\n")
            
            # Non-negativity
            self.steps_text.insert(tk.END, "\nNon negatività:\n")
            for i, (sign, val) in enumerate(zip(self.nn_signs, self.nn_values)):
                if sign.get() != "Libera":
                    self.steps_text.insert(tk.END, f"  {self.variables[i]} {sign.get()} {val.get()}\n")
            
            # Integer constraints
            int_vars = [self.variables[i] for i, var in enumerate(self.integer_vars) if var.get() == 1]
            if int_vars:
                self.steps_text.insert(tk.END, "\nVariabili intere:\n")
                self.steps_text.insert(tk.END, f"  {', '.join(int_vars)}\n")
            
            # Get objective function coefficients
            c = []
            for sign, entry in zip(self.obj_signs, self.obj_coeffs):
                coeff = float(entry.get())
                coeff = coeff if sign.get() == "+" else -coeff
                c.append(coeff)
            
            # Convert to minimization problem if needed
            if self.obj_type.get() == "max":
                c = [-x for x in c]
            
            # Get constraints
            A = []
            b = []
            ineqs = []
            
            for (_, signs, entries, ineq, rhs) in self.constraints:
                row = []
                for sign, entry in zip(signs, entries):
                    val = float(entry.get())
                    val = val if sign.get() == "+" else -val
                    row.append(val)
                A.append(row)
                b.append(float(rhs.get()))
                ineqs.append(ineq.get())
            
            # Add non-negativity constraints
            for i, (sign_box, val_box) in enumerate(zip(self.nn_signs, self.nn_values)):
                sign = sign_box.get()
                val = float(val_box.get())
                
                if sign == "≥":
                    # x_i ≥ val → -x_i ≤ -val
                    row = [0.0] * len(c)
                    row[i] = -1.0
                    A.append(row)
                    b.append(-val)
                    ineqs.append("≤")
                elif sign == "≤":
                    # x_i ≤ val → x_i ≤ val
                    row = [0.0] * len(c)
                    row[i] = 1.0
                    A.append(row)
                    b.append(val)
                    ineqs.append("≤")
                elif sign == "Libera":
                    pass  # No constraint for free variables
            
            # Convert all constraints to standard form (≤)
            A_std = []
            b_std = []
            self.original_constraint_count = len(self.constraints)  # Store for dual variables
            
            for ai, bi, op in zip(A, b, ineqs):
                if op == "≤":
                    A_std.append(ai)
                    b_std.append(bi)
                elif op == "≥":
                    A_std.append([-x for x in ai])
                    b_std.append(-bi)
                elif op == "=":
                    A_std.append(ai)
                    b_std.append(bi)
                    A_std.append([-x for x in ai])
                    b_std.append(-bi)
            
            # Convert to numpy arrays
            c = np.array(c, dtype=float)
            A = np.array(A_std, dtype=float)
            b = np.array(b_std, dtype=float)
            
            # Check for integer variables
            integer_vars = [i for i, var in enumerate(self.integer_vars) if var.get() == 1]
            
            if integer_vars:
                self.steps_text.insert(tk.END, "\n=== PROBLEMA DI PROGRAMMAZIONE LINEARE INTERA ===\n")
                self.steps_text.insert(tk.END, "Risolvo prima il rilassamento continuo:\n")
            
            # Solve using simplex method
            self.run_simplex(c, A, b)
            
            # If integer variables, check if solution is integer
            if integer_vars and self.solution is not None:
                integer_solution = True
                for i in integer_vars:
                    if not np.isclose(self.solution[i], round(self.solution[i])):
                        integer_solution = False
                        break
                
                if not integer_solution:
                    self.steps_text.insert(tk.END, "\nLa soluzione ottima non è intera. Applicare un metodo di branch and bound.\n")
                    self.output_text.insert(tk.END, "\nATTENZIONE: La soluzione ottima non soddisfa i vincoli di interezza.\n")
            
            # Display results
            self.display_results()
            
            # Plot the solution
            self.plot_solution(c, A, b)
            
        except Exception as e:
            self.output_text.insert(tk.END, f"Errore: {str(e)}\n")
            self.steps_text.insert(tk.END, f"\nERRORE: {str(e)}\n")
    
    def run_simplex(self, c, A, b):
        """Implementation of the simplex algorithm with detailed steps"""
        m, n = A.shape
        
        # Record initial tableau
        self.steps_text.insert(tk.END, "\n=== TABLEAU INIZIALE ===\n")
        
        # Add slack variables
        tableau = np.zeros((m+1, n+m+1))
        tableau[:-1, :n] = A
        tableau[:-1, n:n+m] = np.eye(m)
        tableau[:-1, -1] = b
        tableau[-1, :n] = c  # We're always minimizing in standard form
        
        # Display initial tableau
        self.steps_text.insert(tk.END, "\nVariabili in base: ")
        self.steps_text.insert(tk.END, ", ".join([f"s{i+1}" for i in range(m)]) + "\n")
        self.steps_text.insert(tk.END, "Variabili fuori base: ")
        self.steps_text.insert(tk.END, ", ".join([f"x{i+1}" for i in range(n)]) + "\n\n")
        
        self.steps_text.insert(tk.END, self.format_tableau(tableau, n, m))
        self.problem_history.append(("Tableau iniziale", tableau.copy()))
        
        # Bland's rule for anti-cycling
        iteration = 1
        while True:
            # Check for optimality
            if np.all(tableau[-1, :-1] >= -1e-10):
                self.steps_text.insert(tk.END, "\n=== CONDIZIONE DI OTTIMALITÀ SODDISFATTA ===\n")
                self.steps_text.insert(tk.END, "Tutti i coefficienti di costo ridotto sono non negativi.\n")
                break
            
            # Select entering variable (smallest index with negative reduced cost)
            entering = np.argmin(tableau[-1, :-1])
            entering_var = f"x{entering+1}" if entering < n else f"s{entering-n+1}"
            
            self.steps_text.insert(tk.END, f"\n=== ITERAZIONE {iteration} ===\n")
            self.steps_text.insert(tk.END, f"Variabile entrante: {entering_var} (colonna {entering+1})\n")
            self.steps_text.insert(tk.END, f"Motivo: coefficiente di costo ridotto negativo ({tableau[-1, entering]:.4f})\n")
            
            if tableau[-1, entering] >= -1e-10:
                break  # Optimal solution found
            
            # Select leaving variable using minimum ratio test
            ratios = np.where(tableau[:-1, entering] > 1e-10,
                            tableau[:-1, -1] / tableau[:-1, entering], np.inf)
            
            if np.all(ratios == np.inf):
                self.steps_text.insert(tk.END, "\n=== PROBLEMA ILLIMITATO ===\n")
                self.steps_text.insert(tk.END, "Tutti i rapporti sono infiniti. La soluzione ottima è illimitata.\n")
                self.output_text.insert(tk.END, "Problema illimitato.\n")
                return
            
            leaving = np.argmin(ratios)
            leaving_var = f"s{leaving+1}"
            
            self.steps_text.insert(tk.END, f"Variabile uscente: {leaving_var} (riga {leaving+1})\n")
            self.steps_text.insert(tk.END, f"Motivo: rapporto minimo ({ratios[leaving]:.4f})\n")
            self.steps_text.insert(tk.END, f"Elemento pivot: {tableau[leaving, entering]:.4f} (riga {leaving+1}, colonna {entering+1})\n")
            
            # Pivot
            pivot = tableau[leaving, entering]
            tableau[leaving] /= pivot
            
            self.steps_text.insert(tk.END, "\nDopo la normalizzazione della riga pivot:\n")
            self.steps_text.insert(tk.END, self.format_tableau(tableau, n, m))
            self.problem_history.append((f"Iterazione {iteration} - Dopo normalizzazione", tableau.copy()))
            
            for i in range(m + 1):
                if i != leaving:
                    tableau[i] -= tableau[i, entering] * tableau[leaving]
            
            # Update basis variables
            basis_vars = []
            for col in range(n):
                col_vals = tableau[:-1, col]
                if np.sum(np.isclose(col_vals, 1)) == 1 and np.sum(np.isclose(col_vals, 0)) == m-1:
                    basis_vars.append(f"x{col+1}")
            
            for col in range(m):
                col_vals = tableau[:-1, n+col]
                if np.sum(np.isclose(col_vals, 1)) == 1 and np.sum(np.isclose(col_vals, 0)) == m-1:
                    basis_vars.append(f"s{col+1}")
            
            self.steps_text.insert(tk.END, "\nDopo l'operazione di pivot:\n")
            self.steps_text.insert(tk.END, f"Variabili in base: {', '.join(basis_vars)}\n")
            self.steps_text.insert(tk.END, self.format_tableau(tableau, n, m))
            self.problem_history.append((f"Iterazione {iteration} - Dopo pivot", tableau.copy()))
            
            iteration += 1
        
        # Extract solution
        solution = np.zeros(n)
        basis = []
        
        for col in range(n):
            col_vals = tableau[:-1, col]
            if np.sum(np.isclose(col_vals, 1)) == 1 and np.sum(np.isclose(col_vals, 0)) == m-1:
                row = np.argmax(col_vals)
                solution[col] = tableau[row, -1]
                basis.append((row, col))
        
        # Handle alternative solutions
        self.alternative_solutions = []
        for col in range(n):
            if tableau[-1, col] == 0 and col not in [x[1] for x in basis]:
                # Alternative solution exists
                alt_solution = np.zeros(n)
                for row, basis_col in basis:
                    alt_solution[basis_col] = tableau[row, -1]
                alt_solution[col] = 0
                self.alternative_solutions.append(alt_solution)
        
        # Calculate slack variables
        slack_values = b - np.dot(A, solution)
        
        # Calculate optimal value (adjust sign if we converted from maximization)
        optimal_value = tableau[-1, -1]
        if self.obj_type.get() == "max":
            optimal_value = -optimal_value
        
        # Extract dual variables (shadow prices)
        dual_solution = np.zeros(m)
        for row in range(m):
            if n + row < tableau.shape[1] - 1:  # Check if within bounds
                dual_solution[row] = tableau[-1, n + row]
        
        # Adjust dual variables for converted constraints
        final_dual = np.zeros(self.original_constraint_count)
        current_idx = 0
        for i, op in enumerate([x[3].get() for x in self.constraints]):
            if op == "≤":
                final_dual[i] = dual_solution[current_idx]
                current_idx += 1
            elif op == "≥":
                final_dual[i] = -dual_solution[current_idx]
                current_idx += 1
            elif op == "=":
                final_dual[i] = dual_solution[current_idx] - dual_solution[current_idx+1]
                current_idx += 2
        
        self.solution = solution
        self.slack_values = slack_values
        self.optimal_value = optimal_value
        self.dual_solution = final_dual
    
    def format_tableau(self, tableau, n_vars, n_slacks):
        """Format the tableau for display with better alignment"""
        lines = []
        header = ["Base"] + [f"x{i+1}" for i in range(n_vars)] + [f"s{i+1}" for i in range(n_slacks)] + ["RHS"]
        lines.append("  ".join(f"{h:>8}" for h in header))
        
        for i in range(n_slacks):
            row = [f"s{i+1}"]
            for j in range(n_vars + n_slacks + 1):
                row.append(f"{tableau[i, j]:8.4f}")
            lines.append("  ".join(row))
        
        cost_row = ["Cost"]
        for j in range(n_vars + n_slacks + 1):
            cost_row.append(f"{tableau[-1, j]:8.4f}")
        lines.append("  ".join(cost_row))
        
        return "\n".join(lines) + "\n"
    
    def display_results(self):
        """Display the solution in a formatted way"""
        if self.solution is None:
            return
        
        self.output_text.insert(tk.END, "=== SOLUZIONE ===\n\n")
        
        # Display optimal value
        opt_type = "massimo" if self.obj_type.get() == "max" else "minimo"
        self.output_text.insert(tk.END, f"Valore ottimo ({opt_type}): z = {self.optimal_value:.4f}\n\n")
        
        # Display variable values
        self.output_text.insert(tk.END, "Variabili decisionali:\n")
        for i, val in enumerate(self.solution):
            self.output_text.insert(tk.END, f"  {self.variables[i]} = {val:.4f}\n")
        
        # Display slack variables for original constraints
        if len(self.slack_values) > 0:
            self.output_text.insert(tk.END, "\nVariabili di slack:\n")
            for i in range(self.original_constraint_count):
                self.output_text.insert(tk.END, f"  Vincolo {i+1}: s_{i+1} = {self.slack_values[i]:.4f}\n")
        
        # Display dual variables (shadow prices)
        if len(self.dual_solution) > 0:
            self.output_text.insert(tk.END, "\nPrezzi ombra (variabili duali):\n")
            for i, val in enumerate(self.dual_solution):
                self.output_text.insert(tk.END, f"  Vincolo {i+1}: π_{i+1} = {val:.4f}\n")
        
        # Display alternative solutions if they exist
        if len(self.alternative_solutions) > 0:
            self.output_text.insert(tk.END, "\nSoluzioni ottime alternative:\n")
            for alt_sol in self.alternative_solutions:
                sol_str = ", ".join([f"{var} = {val:.4f}" for var, val in zip(self.variables, alt_sol)])
                self.output_text.insert(tk.END, f"  • {sol_str}\n")
        
        self.output_text.insert(tk.END, "\n=== FINE SOLUZIONE ===")
    
    def plot_solution(self, c=None, A=None, b=None):
        """Plot the feasible region and solution"""
        if c is None or A is None or b is None:
            # Get current problem data
            try:
                c = []
                for sign, entry in zip(self.obj_signs, self.obj_coeffs):
                    coeff = float(entry.get())
                    coeff = coeff if sign.get() == "+" else -coeff
                    c.append(coeff)
                
                A = []
                b = []
                ineqs = []
                
                for (_, signs, entries, ineq, rhs) in self.constraints:
                    row = []
                    for sign, entry in zip(signs, entries):
                        val = float(entry.get())
                        val = val if sign.get() == "+" else -val
                        row.append(val)
                    A.append(row)
                    b.append(float(rhs.get()))
                    ineqs.append(ineq.get())
                
                # Add non-negativity constraints
                for i, (sign_box, val_box) in enumerate(zip(self.nn_signs, self.nn_values)):
                    sign = sign_box.get()
                    val = float(val_box.get())
                    
                    if sign == "≥":
                        row = [0.0] * len(c)
                        row[i] = -1.0
                        A.append(row)
                        b.append(-val)
                        ineqs.append("≤")
                    elif sign == "≤":
                        row = [0.0] * len(c)
                        row[i] = 1.0
                        A.append(row)
                        b.append(val)
                        ineqs.append("≤")
                    elif sign == "Libera":
                        pass
                
                # Convert all constraints to standard form (≤)
                A_std = []
                b_std = []
                
                for ai, bi, op in zip(A, b, ineqs):
                    if op == "≤":
                        A_std.append(ai)
                        b_std.append(bi)
                    elif op == "≥":
                        A_std.append([-x for x in ai])
                        b_std.append(-bi)
                    elif op == "=":
                        A_std.append(ai)
                        b_std.append(bi)
                        A_std.append([-x for x in ai])
                        b_std.append(-bi)
                
                # Convert to numpy arrays
                c = np.array(c, dtype=float)
                A = np.array(A_std, dtype=float)
                b = np.array(b_std, dtype=float)
                
            except Exception as e:
                self.ax.clear()
                self.ax.text(0.5, 0.5, "Errore nel tracciamento del grafico", ha="center", va="center")
                self.canvas.draw()
                return
        
        self.ax.clear()
        
        # Only plot if we have 2 variables
        if len(self.variables) != 2:
            self.ax.text(0.5, 0.5, "Grafico disponibile solo per problemi con 2 variabili", 
                        ha="center", va="center")
            self.canvas.draw()
            return
        
        # Determine plot limits based on constraints
        x_min, x_max = 0, 10
        y_min, y_max = 0, 10
        
        # Plot constraints and find feasible region boundaries
        x = np.linspace(x_min, x_max, 400)
        constraint_lines = []
        labels = []
        intersection_points = []
        
        for i in range(len(b)):
            if A[i, 1] != 0:  # Not a vertical line
                y = (b[i] - A[i, 0] * x) / A[i, 1]
                line, = self.ax.plot(x, y, label=f"Vincolo {i+1}")
                constraint_lines.append(line)
                labels.append(f"Vincolo {i+1}")
                
                # Update y limits based on constraint
                valid_y = y[(x >= x_min) & (x <= x_max)]
                if len(valid_y) > 0:
                    y_min = min(y_min, np.min(valid_y))
                    y_max = max(y_max, np.max(valid_y))
                
                # Find intersections with axes
                # x=0 intercept
                y_intercept = b[i] / A[i, 1]
                if y_intercept >= 0:
                    intersection_points.append((0, y_intercept))
                
                # y=0 intercept
                x_intercept = b[i] / A[i, 0]
                if x_intercept >= 0:
                    intersection_points.append((x_intercept, 0))
            else:  # Vertical line (x = const)
                x_val = b[i] / A[i, 0] if A[i, 0] != 0 else 0
                line = self.ax.axvline(x_val, label=f"Vincolo {i+1}")
                constraint_lines.append(line)
                labels.append(f"Vincolo {i+1}")
                
                # Update x limits based on constraint
                x_min = min(x_min, x_val)
                x_max = max(x_max, x_val)
        
        # Find intersections between constraints
        for i in range(len(b)):
            for j in range(i+1, len(b)):
                # Solve the system of equations for the intersection
                A_sys = A[[i, j], :2]
                b_sys = b[[i, j]]
                try:
                    point = np.linalg.solve(A_sys, b_sys)
                    if point[0] >= 0 and point[1] >= 0:
                        intersection_points.append(point)
                except np.linalg.LinAlgError:
                    continue
        
        # Add some padding to limits
        x_min = max(x_min - 1, 0)
        x_max = x_max + 1
        y_min = max(y_min - 1, 0)
        y_max = y_max + 1
        
        # Make plot square by using the larger range
        range_x = x_max - x_min
        range_y = y_max - y_min
        
        if range_x > range_y:
            center_y = (y_min + y_max) / 2
            y_min = center_y - range_x / 2
            y_max = center_y + range_x / 2
        else:
            center_x = (x_min + x_max) / 2
            x_min = center_x - range_y / 2
            x_max = center_x + range_y / 2
        
        # Determine feasible region vertices manually
        vertices = self.find_feasible_vertices(A, b)
        if vertices:
            polygon = Polygon(vertices, closed=True, alpha=0.3, label='Regione ammissibile')
            self.ax.add_patch(polygon)
        
        # Plot optimal solution if it exists
        if self.solution is not None and len(self.solution) == 2:
            self.ax.plot(self.solution[0], self.solution[1], 'ro', markersize=8, label='Soluzione ottima')
            
            # Plot objective function direction and level curves
            if c[1] != 0:
                obj_slope = -c[0] / c[1]
                
                # Main direction line
                obj_line_x = np.array([x_min, x_max])
                obj_line_y = obj_slope * (obj_line_x - self.solution[0]) + self.solution[1]
                self.ax.plot(obj_line_x, obj_line_y, 'g--', linewidth=2, label='Direzione ottimizzazione')
                
                # Level curves
                for level in np.linspace(0.2, 1.0, 4):
                    offset = level * 2
                    level_line_y = obj_slope * (obj_line_x - self.solution[0]) + self.solution[1] + offset
                    self.ax.plot(obj_line_x, level_line_y, 'g:', linewidth=0.5, alpha=0.5)
                    
                    level_line_y = obj_slope * (obj_line_x - self.solution[0]) + self.solution[1] - offset
                    self.ax.plot(obj_line_x, level_line_y, 'g:', linewidth=0.5, alpha=0.5)
                
                # Add gradient vector (perpendicular to level curves)
                vec_length = min(range_x, range_y) * 0.2
                vec_x = vec_length * c[0] / np.linalg.norm(c)
                vec_y = vec_length * c[1] / np.linalg.norm(c)
                
                arrow = FancyArrowPatch((self.solution[0], self.solution[1]),
                                      (self.solution[0] + vec_x, self.solution[1] + vec_y),
                                      color='red', arrowstyle='->', mutation_scale=15,
                                      linewidth=2, label='Vettore gradiente')
                self.ax.add_patch(arrow)
        
        # Plot intersection points
        if intersection_points:
            points = np.array(intersection_points)
            self.ax.scatter(points[:, 0], points[:, 1], color='blue', zorder=5)
            
            # Label points with coordinates
            for point in points:
                self.ax.text(point[0], point[1], f"({point[0]:.1f}, {point[1]:.1f})", 
                           fontsize=8, ha='right', va='bottom',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Draw axes at (0,0)
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ax.axvline(0, color='black', linewidth=0.5)
        
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlabel(self.variables[0])
        self.ax.set_ylabel(self.variables[1])
        self.ax.set_title("Regione Ammissibile")
        
        # Add grid with integer ticks
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Create legend
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(handles, labels, loc='upper right')
        
        self.canvas.draw()
    
    def show_graph_window(self):
        """Show graph in a separate window"""
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Visualizzazione Grafica")
        graph_window.geometry("800x800")
        
        fig = plt.Figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        # Get current problem data
        try:
            c = []
            for sign, entry in zip(self.obj_signs, self.obj_coeffs):
                coeff = float(entry.get())
                coeff = coeff if sign.get() == "+" else -coeff
                c.append(coeff)
            
            A = []
            b = []
            ineqs = []
            
            for (_, signs, entries, ineq, rhs) in self.constraints:
                row = []
                for sign, entry in zip(signs, entries):
                    val = float(entry.get())
                    val = val if sign.get() == "+" else -val
                    row.append(val)
                A.append(row)
                b.append(float(rhs.get()))
                ineqs.append(ineq.get())
            
            # Add non-negativity constraints
            for i, (sign_box, val_box) in enumerate(zip(self.nn_signs, self.nn_values)):
                sign = sign_box.get()
                val = float(val_box.get())
                
                if sign == "≥":
                    row = [0.0] * len(c)
                    row[i] = -1.0
                    A.append(row)
                    b.append(-val)
                    ineqs.append("≤")
                elif sign == "≤":
                    row = [0.0] * len(c)
                    row[i] = 1.0
                    A.append(row)
                    b.append(val)
                    ineqs.append("≤")
                elif sign == "Libera":
                    pass
            
            # Convert all constraints to standard form (≤)
            A_std = []
            b_std = []
            
            for ai, bi, op in zip(A, b, ineqs):
                if op == "≤":
                    A_std.append(ai)
                    b_std.append(bi)
                elif op == "≥":
                    A_std.append([-x for x in ai])
                    b_std.append(-bi)
                elif op == "=":
                    A_std.append(ai)
                    b_std.append(bi)
                    A_std.append([-x for x in ai])
                    b_std.append(-bi)
            
            # Convert to numpy arrays
            c = np.array(c, dtype=float)
            A = np.array(A_std, dtype=float)
            b = np.array(b_std, dtype=float)
            
            # Plot the solution
            self.plot_solution_to_axes(ax, c, A, b)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Errore: {str(e)}", ha="center", va="center")
        
        canvas = FigureCanvasTkAgg(fig, graph_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, graph_window)
        toolbar.update()
    
    def plot_solution_to_axes(self, ax, c, A, b):
        """Plot solution to a given axes object"""
        # Determine plot limits based on constraints
        x_min, x_max = 0, 10
        y_min, y_max = 0, 10
        
        # Plot constraints and find feasible region boundaries
        x = np.linspace(x_min, x_max, 400)
        constraint_lines = []
        labels = []
        intersection_points = []
        
        for i in range(len(b)):
            if A[i, 1] != 0:  # Not a vertical line
                y = (b[i] - A[i, 0] * x) / A[i, 1]
                line, = ax.plot(x, y, label=f"Vincolo {i+1}")
                constraint_lines.append(line)
                labels.append(f"Vincolo {i+1}")
                
                # Update y limits based on constraint
                valid_y = y[(x >= x_min) & (x <= x_max)]
                if len(valid_y) > 0:
                    y_min = min(y_min, np.min(valid_y))
                    y_max = max(y_max, np.max(valid_y))
                
                # Find intersections with axes
                # x=0 intercept
                y_intercept = b[i] / A[i, 1]
                if y_intercept >= 0:
                    intersection_points.append((0, y_intercept))
                
                # y=0 intercept
                x_intercept = b[i] / A[i, 0]
                if x_intercept >= 0:
                    intersection_points.append((x_intercept, 0))
            else:  # Vertical line (x = const)
                x_val = b[i] / A[i, 0] if A[i, 0] != 0 else 0
                line = ax.axvline(x_val, label=f"Vincolo {i+1}")
                constraint_lines.append(line)
                labels.append(f"Vincolo {i+1}")
                
                # Update x limits based on constraint
                x_min = min(x_min, x_val)
                x_max = max(x_max, x_val)
        
        # Find intersections between constraints
        for i in range(len(b)):
            for j in range(i+1, len(b)):
                # Solve the system of equations for the intersection
                A_sys = A[[i, j], :2]
                b_sys = b[[i, j]]
                try:
                    point = np.linalg.solve(A_sys, b_sys)
                    if point[0] >= 0 and point[1] >= 0:
                        intersection_points.append(point)
                except np.linalg.LinAlgError:
                    continue
        
        # Add some padding to limits
        x_min = max(x_min - 1, 0)
        x_max = x_max + 1
        y_min = max(y_min - 1, 0)
        y_max = y_max + 1
        
        # Make plot square by using the larger range
        range_x = x_max - x_min
        range_y = y_max - y_min
        
        if range_x > range_y:
            center_y = (y_min + y_max) / 2
            y_min = center_y - range_x / 2
            y_max = center_y + range_x / 2
        else:
            center_x = (x_min + x_max) / 2
            x_min = center_x - range_y / 2
            x_max = center_x + range_y / 2
        
        # Determine feasible region vertices manually
        vertices = self.find_feasible_vertices(A, b)
        if vertices:
            polygon = Polygon(vertices, closed=True, alpha=0.3, label='Regione ammissibile')
            ax.add_patch(polygon)
        
        # Plot optimal solution if it exists
        if self.solution is not None and len(self.solution) == 2:
            ax.plot(self.solution[0], self.solution[1], 'ro', markersize=8, label='Soluzione ottima')
            
            # Plot objective function direction and level curves
            if c[1] != 0:
                obj_slope = -c[0] / c[1]
                
                # Main direction line
                obj_line_x = np.array([x_min, x_max])
                obj_line_y = obj_slope * (obj_line_x - self.solution[0]) + self.solution[1]
                ax.plot(obj_line_x, obj_line_y, 'g--', linewidth=2, label='Direzione ottimizzazione')
                
                # Level curves
                for level in np.linspace(0.2, 1.0, 4):
                    offset = level * 2
                    level_line_y = obj_slope * (obj_line_x - self.solution[0]) + self.solution[1] + offset
                    ax.plot(obj_line_x, level_line_y, 'g:', linewidth=0.5, alpha=0.5)
                    
                    level_line_y = obj_slope * (obj_line_x - self.solution[0]) + self.solution[1] - offset
                    ax.plot(obj_line_x, level_line_y, 'g:', linewidth=0.5, alpha=0.5)
                
                # Add gradient vector (perpendicular to level curves)
                vec_length = min(range_x, range_y) * 0.2
                vec_x = vec_length * c[0] / np.linalg.norm(c)
                vec_y = vec_length * c[1] / np.linalg.norm(c)
                
                arrow = FancyArrowPatch((self.solution[0], self.solution[1]),
                                      (self.solution[0] + vec_x, self.solution[1] + vec_y),
                                      color='red', arrowstyle='->', mutation_scale=15,
                                      linewidth=2, label='Vettore gradiente')
                ax.add_patch(arrow)
        
        # Plot intersection points
        if intersection_points:
            points = np.array(intersection_points)
            ax.scatter(points[:, 0], points[:, 1], color='blue', zorder=5)
            
            # Label points with coordinates
            for point in points:
                ax.text(point[0], point[1], f"({point[0]:.1f}, {point[1]:.1f})", 
                       fontsize=8, ha='right', va='bottom',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Draw axes at (0,0)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(self.variables[0])
        ax.set_ylabel(self.variables[1])
        ax.set_title("Regione Ammissibile")
        
        # Add grid with integer ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Create legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='upper right')
    
    def find_feasible_vertices(self, A, b):
        """Find vertices of the feasible region for 2D problems"""
        if len(self.variables) != 2:
            return None
        
        # Find all intersection points
        vertices = []
        n = len(b)
        
        # Add axis intercepts
        vertices.append((0, 0))
        
        # Find intersections between constraints
        for i in range(n):
            for j in range(i+1, n):
                a1, a2 = A[i, 0], A[i, 1]
                b1 = b[i]
                c1, c2 = A[j, 0], A[j, 1]
                b2 = b[j]
                
                # Solve the system: a1*x + a2*y = b1, c1*x + c2*y = b2
                det = a1 * c2 - a2 * c1
                
                if abs(det) > 1e-10:  # Not parallel
                    x = (c2 * b1 - a2 * b2) / det
                    y = (a1 * b2 - c1 * b1) / det
                    
                    if x >= -1e-10 and y >= -1e-10:  # Non-negative
                        vertices.append((x, y))
        
        # Filter points that satisfy all constraints
        feasible_vertices = []
        for x, y in vertices:
            feasible = True
            for i in range(n):
                if A[i, 0] * x + A[i, 1] * y > b[i] + 1e-10:
                    feasible = False
                    break
            if feasible:
                feasible_vertices.append((x, y))
        
        # Sort vertices clockwise for proper polygon drawing
        if len(feasible_vertices) > 2:
            center = np.mean(feasible_vertices, axis=0)
            feasible_vertices.sort(key=lambda p: np.arctan2(p[1]-center[1], p[0]-center[0]))
        
        return feasible_vertices
    
    def export_pdf(self):
        """Export the solution and graph to PDF with proper pagination"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")],
            title="Salva soluzione come PDF"
        )
        
        if not filename:
            return
        
        try:
            with PdfPages(filename) as pdf:
                # First page: Problem statement and solution
                fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait in inches
                fig.suptitle("Advanced LP/ILP Solver - Università", fontsize=14, fontweight='bold')
                
                # Create a subplot for text content
                text_ax = fig.add_subplot(111)
                text_ax.axis('off')
                
                # Problem statement
                content = []
                content.append(("=== PROBLEMA INIZIALE ===\n\n", 12, True))
                
                # Objective function
                obj_str = f"{self.obj_type.get()} z = "
                for i, (sign, entry) in enumerate(zip(self.obj_signs, self.obj_coeffs)):
                    coeff = float(entry.get())
                    coeff = coeff if sign.get() == "+" else -coeff
                    if i > 0 and coeff >= 0:
                        obj_str += "+ "
                    obj_str += f"{coeff} {self.variables[i]} "
                content.append((obj_str + "\n\n", 10, False))
                
                # Constraints
                content.append(("Vincoli:\n", 10, False))
                for i, (_, signs, entries, ineq, rhs) in enumerate(self.constraints):
                    constr_str = ""
                    for j, (sign, entry) in enumerate(zip(signs, entries)):
                        val = float(entry.get())
                        val = val if sign.get() == "+" else -val
                        if j > 0 and val >= 0:
                            constr_str += "+ "
                        constr_str += f"{val} {self.variables[j]} "
                    constr_str += f"{ineq.get()} {rhs.get()}"
                    content.append((f"  {constr_str}\n", 10, False))
                
                # Non-negativity
                content.append(("\nNon negatività:\n", 10, False))
                for i, (sign, val) in enumerate(zip(self.nn_signs, self.nn_values)):
                    if sign.get() != "Libera":
                        content.append((f"  {self.variables[i]} {sign.get()} {val.get()}\n", 10, False))
                
                # Integer constraints
                int_vars = [self.variables[i] for i, var in enumerate(self.integer_vars) if var.get() == 1]
                if int_vars:
                    content.append(("\nVariabili intere:\n", 10, False))
                    content.append((f"  {', '.join(int_vars)}\n", 10, False))
                
                # Solution
                if self.solution is not None:
                    content.append(("\n=== SOLUZIONE ===\n\n", 12, True))
                    
                    # Optimal value
                    opt_type = "massimo" if self.obj_type.get() == "max" else "minimo"
                    content.append((f"Valore ottimo ({opt_type}): z = {self.optimal_value:.4f}\n\n", 10, False))
                    
                    # Variables
                    content.append(("Variabili decisionali:\n", 10, False))
                    for i, val in enumerate(self.solution):
                        content.append((f"  {self.variables[i]} = {val:.4f}\n", 10, False))
                    
                    # Slack variables
                    if len(self.slack_values) > 0:
                        content.append(("\nVariabili di slack:\n", 10, False))
                        for i in range(self.original_constraint_count):
                            content.append((f"  Vincolo {i+1}: s_{i+1} = {self.slack_values[i]:.4f}\n", 10, False))
                    
                    # Dual variables
                    if len(self.dual_solution) > 0:
                        content.append(("\nPrezzi ombra (variabili duali):\n", 10, False))
                        for i, val in enumerate(self.dual_solution):
                            content.append((f"  Vincolo {i+1}: π_{i+1} = {val:.4f}\n", 10, False))
                
                # Render text content with proper pagination
                y_position = 0.95
                page_content = []
                
                for text, fontsize, is_bold in content:
                    # Calculate required space
                    lines = text.count('\n') + 1
                    space_needed = lines * 0.04
                    
                    # If we don't have enough space, create a new page
                    if y_position - space_needed < 0.05:
                        # Add current content to page
                        for line, fs, bold in page_content:
                            text_ax.text(0.05, y_pos, line, 
                                        fontsize=fs, 
                                        weight='bold' if bold else 'normal',
                                        transform=text_ax.transAxes,
                                        verticalalignment='top')
                            y_pos -= 0.04 * (fs/10)  # Adjust line spacing based on font size
                        
                        pdf.savefig(fig)
                        plt.close(fig)
                        
                        # Create new page
                        fig = plt.figure(figsize=(8.27, 11.69))
                        fig.suptitle("Advanced LP/ILP Solver - Università (continua)", fontsize=14, fontweight='bold')
                        text_ax = fig.add_subplot(111)
                        text_ax.axis('off')
                        y_position = 0.95
                        page_content = []
                    
                    # Add to current page content
                    page_content.append((text, fontsize, is_bold))
                    y_position -= space_needed
                
                # Add remaining content to last page
                if page_content:
                    y_pos = 0.95
                    for line, fs, bold in page_content:
                        text_ax.text(0.05, y_pos, line, 
                                    fontsize=fs, 
                                    weight='bold' if bold else 'normal',
                                    transform=text_ax.transAxes,
                                    verticalalignment='top')
                        y_pos -= 0.04 * (fs/10)
                    
                    pdf.savefig(fig)
                    plt.close(fig)
                
                # Second page: Graph
                fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
                ax = fig.add_subplot(111)
                
                # Get current problem data
                try:
                    c = []
                    for sign, entry in zip(self.obj_signs, self.obj_coeffs):
                        coeff = float(entry.get())
                        coeff = coeff if sign.get() == "+" else -coeff
                        c.append(coeff)
                    
                    A = []
                    b = []
                    ineqs = []
                    
                    for (_, signs, entries, ineq, rhs) in self.constraints:
                        row = []
                        for sign, entry in zip(signs, entries):
                            val = float(entry.get())
                            val = val if sign.get() == "+" else -val
                            row.append(val)
                        A.append(row)
                        b.append(float(rhs.get()))
                        ineqs.append(ineq.get())
                    
                    # Add non-negativity constraints
                    for i, (sign_box, val_box) in enumerate(zip(self.nn_signs, self.nn_values)):
                        sign = sign_box.get()
                        val = float(val_box.get())
                        
                        if sign == "≥":
                            row = [0.0] * len(c)
                            row[i] = -1.0
                            A.append(row)
                            b.append(-val)
                            ineqs.append("≤")
                        elif sign == "≤":
                            row = [0.0] * len(c)
                            row[i] = 1.0
                            A.append(row)
                            b.append(val)
                            ineqs.append("≤")
                        elif sign == "Libera":
                            pass
                    
                    # Convert all constraints to standard form (≤)
                    A_std = []
                    b_std = []
                    
                    for ai, bi, op in zip(A, b, ineqs):
                        if op == "≤":
                            A_std.append(ai)
                            b_std.append(bi)
                        elif op == "≥":
                            A_std.append([-x for x in ai])
                            b_std.append(-bi)
                        elif op == "=":
                            A_std.append(ai)
                            b_std.append(bi)
                            A_std.append([-x for x in ai])
                            b_std.append(-bi)
                    
                    # Convert to numpy arrays
                    c = np.array(c, dtype=float)
                    A = np.array(A_std, dtype=float)
                    b = np.array(b_std, dtype=float)
                    
                    # Plot the solution
                    self.plot_solution_to_axes(ax, c, A, b)
                    ax.set_title("Regione Ammissibile", pad=20)
                    
                except Exception as e:
                    ax.text(0.5, 0.5, f"Errore nel tracciamento del grafico: {str(e)}", ha="center", va="center")
                
                pdf.savefig(fig)
                plt.close(fig)
                
                # Additional pages for solution steps if needed
                steps_text = self.steps_text.get(1.0, tk.END)
                if steps_text.strip():
                    # Split steps into pages
                    lines = steps_text.split('\n')
                    current_page_lines = []
                    lines_per_page = 80  # Approximate number of lines that fit on a page
                    
                    for i, line in enumerate(lines):
                        current_page_lines.append(line)
                        
                        if len(current_page_lines) >= lines_per_page or i == len(lines)-1:
                            fig = plt.figure(figsize=(8.27, 11.69))
                            fig.suptitle("Passaggi della Soluzione", fontsize=14, fontweight='bold')
                            text_ax = fig.add_subplot(111)
                            text_ax.axis('off')
                            
                            y_pos = 0.95
                            for l in current_page_lines:
                                text_ax.text(0.05, y_pos, l, 
                                            fontsize=9, 
                                            family='monospace',
                                            transform=text_ax.transAxes,
                                            verticalalignment='top')
                                y_pos -= 0.025
                            
                            pdf.savefig(fig)
                            plt.close(fig)
                            current_page_lines = []
                
            messagebox.showinfo("Successo", f"PDF salvato con successo:\n{filename}")
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile salvare il PDF:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LinearProgrammingSolver(root)
    root.mainloop()