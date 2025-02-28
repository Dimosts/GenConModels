import json
import random
import glob
import os

import cpmpy as cp
import numpy as np
from cpmpy import *
from cpmpy.expressions.utils import all_pairs
from cpmpy.transformations.normalize import toplevel_list
import pickle
from GenAcq import GenAcq
from utils.generic_utils import combine_sets_distinct, get_combinations, get_divisors, get_variables_from_constraints
from cpmpy.expressions.core import Operator
from utils.experiment_utils import load_instances
from ProblemInstance import ProblemInstance

def construct_benchmarks():

    sudoku_params = set()
    while len(sudoku_params) < 10:  # Generate 10 unique Sudoku instances
        grid_size = random.randint(4, 16)  # Random grid size between 4 and 16
        while len(get_divisors(grid_size)) == 0:
            grid_size = random.randint(4, 16)  # Random grid size between 4 and 16
        block_size_row = random.randint(2, (grid_size // 2))  # Random row block size

        # Ensure the block sizes multiply to the grid size
        while grid_size % block_size_row != 0:
            block_size_row = random.randint(2, (grid_size // 2))
        block_size_col = grid_size // block_size_row

        params = (block_size_row, block_size_col, grid_size)
        if params not in sudoku_params:
            sudoku_params.add(params)
            construct_sudoku(*params)


    # Construct golomb instances:
    marks = range(5,15)
    for m in marks:
        construct_golomb(m)

    nurse_params = set()
    while len(nurse_params) < 10:  # Generate 10 unique NR instances
        #construct nurse rostering instances
        shifts_per_day = random.randint(3, 4)  #  3 or 4 shifts per day
        num_days = random.randint(7, 30)  # 1 to 4 weeks
        num_nurses = random.randint(12, 20)  # 12 to 20 nurses
        nurses_per_shift = random.randint(3, num_nurses // shifts_per_day)  # Ensure it's feasible

        params = (shifts_per_day, num_days, num_nurses, nurses_per_shift)
        if params not in nurse_params:
            nurse_params.add(params)
            construct_nurse_rostering(shifts_per_day, num_days, num_nurses, nurses_per_shift)

    exam_params = set()
    while len(exam_params) < 10:  # Generate 10 unique ET instances

        # Define reasonable ranges for each parameter
        NSemesters = random.randint(4, 10)  # 4 to 10 semesters
        courses_per_semester = random.randint(3, 7)  # 3 to 10 courses per semester
        rooms = random.randint(1, 2)  # 1 to 2 rooms
        timeslots_per_day = random.randint(2, 6)  # 2 to 5 timeslots per day
        total_courses = courses_per_semester*NSemesters # total courses
        total_timeslots = timeslots_per_day*rooms # total timeslots
        min_days = total_courses // total_timeslots # min days to be SAT
        days_for_exams = random.randint(min_days, min_days+5)  # range of 5 days

        params = (NSemesters, courses_per_semester, rooms, timeslots_per_day, days_for_exams)
        if params not in exam_params:
            exam_params.add(params)
            construct_examtt(NSemesters, courses_per_semester, rooms, timeslots_per_day, days_for_exams)

# Benchmark construction
def construct_sudoku(block_size_row, block_size_col, grid_size):
    # Variables
    grid = intvar(1, grid_size, shape=(grid_size, grid_size), name="grid")

    model = Model()

    partitions = []

    # Constraints on rows and columns
    for row in grid:
        model += AllDifferent(row).decompose()

    for col in grid.T:  # numpy's Transpose
        model += AllDifferent(col).decompose()

    # Constraints on blocks
    for i in range(0, grid_size, block_size_row):
        for j in range(0, grid_size, block_size_col):
            model += AllDifferent(grid[i:i + block_size_row, j:j + block_size_col]).decompose()  # python's indexing
            partitions.append(grid[i:i + block_size_row, j:j + block_size_col])

    C = list(model.constraints)
    C_T = set(toplevel_list(C))

    # Create a dictionary with the parameters
    params = {"block_size_row": block_size_row, "block_size_col": block_size_col, "grid_size": grid_size}

    output_file = f"benchmarks/sudoku_{grid_size}_{block_size_row}_{block_size_col}"

    # Write the target set to a pickle file
    with open(f"{output_file}.pickle", 'wb') as pickle_file:
        pickle.dump(C_T, pickle_file)

    # Write the parameters to a JSON file
    with open(f"{output_file}.json", 'w') as json_file:
        json.dump(params, json_file)

    # Write the partitions to a pickle file
    with open(f"{output_file}.par", 'wb') as pickle_file:
        pickle.dump(partitions, pickle_file)


def construct_nurse_rostering(shifts_per_day, num_days, num_nurses, nurses_per_shift):
    # Define the variables
    roster_matrix = intvar(1, num_nurses, shape=(num_days, shifts_per_day, nurses_per_shift), name="shifts")

    # Define the constraints
    model = Model()

    # Constraint: Each shift in a day must be assigned to a different nurse
    for day in range(num_days):
        model += AllDifferent(roster_matrix[day, ...]).decompose()

    # Constraint: The last shift of a day cannot have the same nurse as the first shift of the next day
    for day in range(num_days - 1):
        model += AllDifferent(roster_matrix[day, shifts_per_day - 1], roster_matrix[day + 1, 0]).decompose()

    if model.solve():
        print("solution exists")
    else:
        print("no solution")
        raise Exception("The problem has no solution")

    C = list(model.constraints)

    # it is needed to be able to make the "suboracle" in ask_query ... Will fix later in a better way.
    C_T = set(toplevel_list(C))

    output_file = f"benchmarks/nurse_rostering_adv_{shifts_per_day}_{num_days}_{num_nurses}_{nurses_per_shift}_{len(C_T)}"

    # Create a dictionary with the parameters
    params = {"shifts_per_day": shifts_per_day, "num_days": num_days, "num_nurses": num_nurses, "nurses_per_shift": nurses_per_shift}

    # Write the target set to a pickle file
    with open(f"{output_file}.pickle", 'wb') as pickle_file:
        pickle.dump(C_T, pickle_file)

    # Write the parameters to a JSON file
    with open(f"{output_file}.json", 'w') as json_file:
        json.dump(params, json_file)


def construct_examtt(NSemesters=10, courses_per_semester=6, rooms=5, timeslots_per_day=3, days_for_exams=15):
    total_courses = NSemesters * courses_per_semester
    slots_per_day = rooms * timeslots_per_day
    total_slots = slots_per_day * days_for_exams

    # Variables
    courses = intvar(1, total_slots, shape=(NSemesters, courses_per_semester), name="courses")
    all_courses = courses.flatten()

    model = Model()

    model += AllDifferent(all_courses).decompose()

    # Constraints on courses of same semester
    for row in courses:
        model += AllDifferent(row // slots_per_day).decompose()

    C = list(model.constraints)

    print(model)

    if model.solve():
        print(courses.value())
    else:
        print("no solution")

    # it is needed to be able to make the "suboracle" in ask_query
    C_T = set(toplevel_list(C))

    output_file = f"benchmarks/exam_timetabling_{NSemesters}_{courses_per_semester}_{rooms}_{timeslots_per_day}_{days_for_exams}_{len(C_T)}"

    # Create a dictionary with the parameters
    params = {"NSemesters": NSemesters, "courses_per_semester": courses_per_semester,
              "timeslots_": timeslots_per_day*rooms, "days_for_exams": days_for_exams}

    # Write the target set to a pickle file
    with open(f"{output_file}.pickle", 'wb') as pickle_file:
        pickle.dump(C_T, pickle_file)

    # Write the parameters to a JSON file
    with open(f"{output_file}.json", 'w') as json_file:
        json.dump(params, json_file)


def construct_golomb(marks):
    # Variables
    grid = intvar(1, marks*5, shape=(1, marks), name="grid")

    model = Model()

    combinations = get_combinations(grid, 2)
    result_combinations = combine_sets_distinct(combinations, combinations)
    for ((v1, v2), (v3, v4)) in result_combinations:
        model += abs(v1 - v2) != abs(v3 - v4)

    for i in range(marks-1):
        model += grid[0, i] < grid[0, i+1]

    C_T = list(model.constraints)

    # Create a dictionary with the parameters
    params = {"marks": marks}

    output_file = f"benchmarks/{marks}golomb"

    # Write the target set to a pickle file
    with open(f"{output_file}.pickle", 'wb') as pickle_file:
        pickle.dump(C_T, pickle_file)

    # Write the parameters to a JSON file
    with open(f"{output_file}.json", 'w') as json_file:
        json.dump(params, json_file)

def create_benchmark_datasets():
    # Process all benchmark types
    benchmarks = ['sudoku', 'golomb', 'nurse_rostering', 'exam_timetabling']
    
    for benchmark in benchmarks:
        # Create and clean dataset directory
        dataset_dir = f"benchmarks/{benchmark}/datasets"
        if os.path.exists(dataset_dir):
            # Remove all existing dataset files
            for file in os.listdir(dataset_dir):
                if file.endswith(".pickle"):
                    os.remove(os.path.join(dataset_dir, file))
            print(f"Cleaned existing datasets in {dataset_dir}")
        else:
            os.makedirs(dataset_dir)
            print(f"Created new dataset directory: {dataset_dir}")
        
        # Load all instances for the given benchmark
        instances = load_instances(benchmark)
        
        if not instances:
            print(f'No instances found for benchmark {benchmark}')
            continue
        
        # Process all instances
        for instance in instances:
            CSP = instance['csp']
            params = instance['params']
            
            # Create dataset directory if it doesn't exist
            dataset_dir = f"benchmarks/{benchmark}/datasets"
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Create a ProblemInstance object with the CSP and parameters
            problem_instance = ProblemInstance(
                constraints=CSP.copy(),
                params=params.copy(),
                name=f"{benchmark}_input",
            )
            
            # Generate the dataset using GenAcq
            ga = GenAcq(inputInstances=problem_instance)
            ga.create_input_dataset()
            datasetX, datasetY = ga.get_dataset()
            
            # Save the datasets using the same format as the pickle files
            out_file_X = f"{dataset_dir}/{instance['name']}_X.pickle"
            out_file_Y = f"{dataset_dir}/{instance['name']}_Y.pickle"
            
            with open(out_file_X, 'wb') as handle:
                pickle.dump(datasetX, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(out_file_Y, 'wb') as handle:
                pickle.dump(datasetY, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"Created dataset for {benchmark} with {len(CSP)} constraints")

def recreate_benchmarks():
    # Get all JSON files in the benchmarks directory
    json_files = glob.glob("benchmarks/*/*.json")
    
    for json_file in json_files:
        # Load parameters from JSON file
        with open(json_file, 'r') as f:
            params = json.load(f)
        
        # Determine benchmark type from filename
        filename = os.path.basename(json_file)
        if filename.startswith('sudoku'):
            construct_sudoku(
                params['block_size_row'],
                params['block_size_col'],
                params['grid_size']
            )
        elif filename.startswith('nurse_rostering'):
            construct_nurse_rostering(
                params['shifts_per_day'],
                params['num_days'],
                params['num_nurses'],
                params['nurses_per_shift']
            )
        elif filename.startswith('exam_timetabling'):
            # Parse parameters from filename
            # Format: exam_timetabling_NSemesters_courses_per_semester_rooms_timeslots_per_day_days_for_exams_constraints
            parts = filename.replace('.json', '').split('_')
            construct_examtt(
                NSemesters=int(parts[2]),
                courses_per_semester=int(parts[3]),
                rooms=int(parts[4]),
                timeslots_per_day=int(parts[5]),
                days_for_exams=int(parts[6])
            )
        elif filename.endswith('golomb.json'):
            construct_golomb(params['marks'])
        else:
            print(f"Unknown benchmark type for file: {json_file}")

if __name__ == "__main__":
    # construct_benchmarks()
    create_benchmark_datasets()
    #recreate_benchmarks()  # Add this line to run the recreation