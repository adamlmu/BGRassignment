Assignment 2: Path Planning - Optimized Path Planning Using Evolutionary Algorithm

**Personal note:**
I tried to solve the solution with the help of an evolutionary algorithm and its principles, but unfortunately I didn't have a lot of time to bring it to maximum optimization. 
Hope you can see the vision and the general idea with the performance I have presented here

## **Overview**
This project presents an approach to optimize a smooth path within a race track layout using an **evolutionary algorithm**. The algorithm outputs a path that minimizes curvature and length while staying inside of the track boundaries defined by the position of cones. The process involves

## **Requirements**
- **Operating System**: Windows
- **Python**: Version 3.8 or higher
- Required Python libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `math`
  - `random`
- `scipy`
  - `deap`

## **Installation**
1. Clone or download the repository.
2. Make sure Python is installed on your Windows system.
3. Install the needed Python libraries with the following command:
   ```bash
   pip install pandas numpy matplotlib scipy deap
   ```
4. Put the `BrandsHatchLayout.csv` file in the same directory as the script.

## **Usage**
1. Open a terminal or command prompt in the script's directory.
2. Run the script:
   ```bash
   python optimized_path_planner.py
   ```
3. After running:
   - The code will optimize the path and display the result.
   - It will plot the inner and outer track boundaries, cones, and the optimized path.
