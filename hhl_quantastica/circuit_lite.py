#
# Lightweight single-file quantum circuit and simple & naive simulator without external dependencies.
# With this, you can assemble and run small and simple quantum circuits without large QC framework.
#
# Made by Quantastica 2022
#

import os
import json
import re
import numpy as np
from copy import deepcopy


gate_defs_path = os.path.join(os.path.abspath(""), "gate_defs.json")
with open(gate_defs_path, encoding="utf-8") as file:
    gate_defs = json.load(file)


class CircuitLite:
    def __init__(self):
        self.params = []
        self.program = []
        self.subroutines = []

    def clear(self):
        self.params = []
        self.program = []
        self.subroutines = []


    def num_qubits(self):
        max_qubit = 0
        for instruction in self.program:
            if("wires" in instruction):
                max_qubit = max(max_qubit, max(instruction["wires"]))

        return max_qubit + 1

    # Add gate to the circuit
    def add(self, gate_name, target_qubits, gate_params=[]):
        gate_options = { "params": {}, "condition": {}, "creg": {} }
        
        # Check gate name and get gate definition
        gate_def = None
        subroutine = None
        if(gate_name in gate_defs):
            gate_def = gate_defs[gate_name]
        else:
            subroutine = self.get_subroutine(gate_name)

        if(gate_def is None and subroutine is None):
            raise Exception("Unknown gate \"" + gate_name + "\"")
        
        # Check target qubits
        wires = []
        if(not isinstance(target_qubits, (list, np.ndarray))):
            wires.append(target_qubits)
        else:
            wires = target_qubits.tolist() if isinstance(target_qubits, np.ndarray) else target_qubits
            
        gate_qubits = 0
        if(gate_def is not None):
            if("matrix" in gate_def):
                if(len(gate_def["matrix"]) > 0):
                    gate_qubits = int(np.log2(len(gate_def["matrix"])))
                else:
                    raise Exception("Gate \"" + gate_name + "\" is not supported.")
            else:
                raise Exception("Gate \"" + gate_name + "\" is not supported.")
        else:
            if(subroutine is not None):
                gate_qubits = subroutine.num_qubits()
            else:
                raise Exception("Unknown gate \"" + gate_name + "\".")

        if(len(wires) != gate_qubits):
            raise Exception("Invalid number of target qubits for gate \"" + gate_name + "\". Expecting " + str(gate_qubits) + " qubits.")


        # Check params
        def_params = []
        if(gate_def is not None and "params" in gate_def):
            def_params = gate_def["params"]
        else:
            if(subroutine is not None):
                def_params = subroutine.params

        params = []
        
        if(not isinstance(gate_params, (list, np.ndarray))):
            if(isinstance(gate_params, dict)):
                for param_name in def_params:
                    if(param_name in gate_params):
                        params.append(gate_params[param_name])
                    else:
                        ## !!!
                        print(gate_params)
                        ## !!!
                        raise Exception("Parameter \"" + param_name + "\" not found in gate \"" + gate_name + "\"")
            else:
                params.append(gate_params)
        else:
            params = gate_params.tolist() if isinstance(gate_params, np.ndarray) else gate_params

        if(len(def_params) > 0):
            if(len(def_params) != len(params)):
                raise Exception("Invalid number of params for gate \"" + gate_name + "\". Expecting " + str(len(def_params)) + " params.")

            for param_index in range(len(def_params)):
                param_name = def_params[param_index]                
                gate_options["params"][param_name] = params[param_index]
        else:
            if(len(params) > 0):
                raise Exception("Gate \"" + gate_name + "\" doesn't expect any params.")
        
        self.program.append({ "name": gate_name, "wires": wires, "options": gate_options })


    def register_subroutine(self, name, subroutine):
        self.subroutines.append({ "name": name, "circuit": subroutine })


    def get_subroutine(self, name):
        for subroutine_def in self.subroutines:
            if(subroutine_def["name"] == name):
                return subroutine_def["circuit"]

        return None

    # Import circuit from QubitToaster format
    def from_toaster(self, toaster_dict):
        self.clear()
        
        for instruction in toaster_dict["program"]:
            gate_options = {}
            if("options" in instruction):
                gate_options = deepcopy(instruction["options"])

            if("params" not in gate_options):
                gate_options["params"] = {}
                
            if("condition" not in gate_options):
                gate_options["condition"] = {}
            
            if("creg" not in gate_options):
                gate_options["creg"] = {}

            self.add(instruction["name"], instruction["wires"], gate_options["params"])


    def to_toaster(self, global_params={}):
        circuit = self.decompose(inplace=False)
        toaster = { "qubits": circuit.num_qubits(), "program": [], "cregs": [] }

        for gate in circuit.program:
            gate_name = gate["name"]
            gate_wires = deepcopy(gate["wires"])
            gate_options = deepcopy(gate["options"])
            
            gate_def = gate_defs[gate_name]
            raw_matrix = gate_def["matrix"]
            gate_matrix = self.eval_matrix(raw_matrix, gate_options["params"], global_params)
            
            toaster["program"].append({ "name": gate_name, "wires": gate_wires, "matrix": gate_matrix, "options": gate_options })
 
        return toaster

        
    # Export to OpenQASM 2.0
    def qasm(self, as_gate="", global_params={}, options={}):
        no_wires = options["no_wires"] if "no_wires" in options else False
        no_params = options["no_params"] if "no_params" in options else False
        eval_params = options["eval_params"] if "eval_params" in options else True
        decimal_places = options["decimal_places"] if "decimal_places" in options else 8

        num_qubits = self.num_qubits()
        
        indent = ""
        if(as_gate != ""):
            indent = "\t"

        lines = []
                
        for gate in self.program:
            s = ""
            s += indent
            s += gate["name"]

            if(not no_params):
                if("options" in gate and "params" in gate["options"]):
                    param_str = ""
                    for param in gate["options"]["params"].values():
                        if(len(param_str)):
                            param_str += ", "

                        if(isinstance(param, str)):
                            if(eval_params and as_gate == ""):
                                param = self.eval_expression(param, global_params)
                                param_str += str(np.round(param, decimal_places))
                            else:
                                param_str += param
                        else:
                            param_str += str(np.round(param, decimal_places))

                    if(len(param_str)):
                        s += " (" + param_str + ")"

            if(not no_wires):
                if("wires" in gate):
                    wire_str = ""
                    for wire in gate["wires"]:
                        if(len(wire_str)):
                            wire_str += ", "
                        if(as_gate != ""):
                            wire_str += "q" + str(wire)
                        else:
                            wire_str += "q[" + str(wire) + "]"
                    s += " " + wire_str
            s += ";"
            lines.append(s)

        s = ""
        
        for subroutine_def in self.subroutines:
            subroutine = subroutine_def["circuit"]
            sub_qasm = subroutine.qasm(as_gate=subroutine_def["name"])
            s += sub_qasm + "\n"
        
        if(as_gate != ""):
            s += "gate " + as_gate
            
            if(len(self.params) > 0):
                s += " ("
                for i in range(len(self.params)):
                    if(i > 0):
                        s += ", "
                    s += self.params[i]
                s += ")"
            
            for i in range(num_qubits):
                if(i > 0):
                    s += ","
                s += " q" + str(i)
            s += "\n{\n"
        else:
            s = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[" + str(num_qubits) + "];\n" + s

        qasm_str = s + "\n".join(lines)

        if(as_gate != ""):
            qasm_str += "\n}\n"

        return qasm_str


    # Get rid of composite gates
    def decompose(self, inplace=False):
        circuit = CircuitLite()

        circuit.params = deepcopy(self.params)
        
        for instruction in self.program:
            subroutine = self.get_subroutine(instruction["name"])
            if(subroutine is None):
                circuit.program.append(deepcopy(instruction))
            else:
                decomposed_subroutine = subroutine.decompose(inplace=False)
                for sub_instruction in decomposed_subroutine.program:
                    # rewire
                    for wire_index in range(len(sub_instruction["wires"])):
                        sub_instruction["wires"][wire_index] = instruction["wires"][sub_instruction["wires"][wire_index]]
                    circuit.program.append(sub_instruction)
        
        return circuit
    

    def eval_expression(self, s, params):
        escaped_string = re.sub(r'\blambda\b', '_lambda', s)
        
        escaped_params = {}
        for param_name in params:
            escaped_param_name = re.sub(r'\blambda\b', '_lambda', param_name)
            escaped_params[escaped_param_name] = params[param_name]
                
        return eval(escaped_string, { "i": 1j, "pi": np.pi, "PI": np.pi, "e": np.e, "sin": np.sin, "cos": np.cos, "exp": np.exp, "sqrt": np.sqrt }, escaped_params)

    
    def eval_matrix(self, unitary, gate_params, global_params):
        new_unitary = []
        for row in unitary:
            new_row = []
            for cell in row:
                if(isinstance(cell, str)):
                    # Evaluate expression
                    params = {}
                    for param_name in gate_params.keys():
                        param_value = gate_params[param_name]
                        if(isinstance(param_value, str)):
                            # Global params
                            param_val = self.eval_expression(param_value, global_params)
                            params[param_name] = param_val
                        else:
                            params[param_name] = param_value

                    value = self.eval_expression(cell, params)
                    new_row.append(value)
                else:
                    new_row.append(cell)

            new_unitary.append(new_row)

        return new_unitary


            
#
# Naive simulator. Suitable for up to 10 qubits. Spends memory and CPU like crazy.
#

class SimulatorLite:

    def __init__(self):
        self.state = []
    

    def clear(self):
        self.state = []


    def init_state(self, num_qubits):
        vector_length = 2**num_qubits
        self.state = np.zeros(vector_length, dtype=complex)
        self.state[0] = 1+0j

    
    def expand_matrix(self, total_qubits, gate_unitary, target_qubits, reverse_bits):
        target_wires = []
        for target_qubit in target_qubits:
            if(reverse_bits is True):
                target_wires.append(target_qubit)
            else:
                target_wires.append(total_qubits - target_qubit - 1)

        target_wires.reverse()

        operator_dimension = 2**total_qubits
        operator = np.zeros((operator_dimension, operator_dimension), dtype=complex)

        reset_mask = 0
        inverse_mask = 0
        for wire in range(total_qubits):
            if(wire in target_wires):
                reset_mask |= 2**wire
            else:
                inverse_mask |= 2**wire

        gate_dimension = len(gate_unitary)
        for gate_row_index in range(gate_dimension):
            row_mask = 0
            for gate_bit in range(len(target_wires)):
                if(gate_row_index & 2**gate_bit != 0):
                    row_mask |= 2**target_wires[gate_bit]

            for gate_col_index in range(gate_dimension):
                col_mask = 0
                for gate_bit in range(len(target_wires)):
                    if(gate_col_index & 2**gate_bit != 0):
                        col_mask |= 2**target_wires[gate_bit]

                gate_element = gate_unitary[gate_row_index][gate_col_index]

                for operator_row in range(operator_dimension):
                    masked_row = operator_row & reset_mask
                    if(masked_row == row_mask):
                        for operator_col in range(operator_dimension):
                            if(operator_col & inverse_mask == operator_row & inverse_mask):
                                masked_col = operator_col & reset_mask

                                if(masked_col == col_mask):
                                    operator[operator_row][operator_col] = gate_element

        return operator

    
    def execute_gate(self, num_qubits, matrix, target_qubits, reverse_bits):
        operator = self.expand_matrix(num_qubits, matrix, target_qubits, reverse_bits)
        self.state = np.dot(operator, self.state)

    #
    # Execute quantum circuit. Resulting wavefuction is in self.state. Simple as that.
    # if "reverse_bits" is True, it will behave as Qiskit (little endian)
    #
    def execute(self, circuit, global_params={}, reverse_bits=False):
        toaster_circuit = circuit.to_toaster(global_params)
        program = toaster_circuit["program"]
        
        num_qubits = circuit.num_qubits()

        self.init_state(num_qubits)
        
        for gate in program:
            matrix = gate["matrix"]
            target_qubits = gate["wires"]
            
            # Apply gate
            self.execute_gate(num_qubits, matrix, target_qubits, reverse_bits)
