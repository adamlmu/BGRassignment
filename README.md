# BGRassignment

# Optimized Path Planning Using Evolutionary Algorithm

## **Overview**
This project demonstrates an approach to optimize a smooth path within a race track layout using an **evolutionary algorithm**. The algorithm generates a path that minimizes curvature and length while adhering to the track boundaries defined by cone positions. The process involves:
1. **Data Preprocessing**: Reading cone positions from a CSV file and defining track limits.
2. **Spline Generation**: Creating smooth inner, outer, and middle track boundaries.
3. **Evolutionary Optimization**: Using a genetic algorithm (via DEAP library) to optimize the path based on length, smoothness, and adherence to track limits.

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
2. Ensure Python is installed on your Windows system.
3. Install the required Python libraries using the following command:
   ```bash
   pip install pandas numpy matplotlib scipy deap
   ```
4. Place the `BrandsHatchLayout.csv` file in the same directory as the script.

## **Usage**
1. Open a terminal or command prompt in the script's directory.
2. Run the script:
   ```bash
   python optimized_path_planner.py
   ```
3. Upon execution:
   - The program will generate and visualize the optimized path.
   - A plot will be displayed showing the inner and outer track boundaries, cones, and the optimized path.
