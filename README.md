# Computing the Optimal Road Trip Across the U.S.

This project provides the methodology and code for computing the optimal road trip across the United States using a genetic algorithm. The code is implemented in Python and is designed to allow users to input their own list of waypoints and generate an optimized driving route.

**Author:** Rajarshi Somvanshi

---

## Overview

This repository contains an end-to-end solution for planning a road trip that visits a list of user-specified destinations in the most efficient order possible. By leveraging the Google Maps API and genetic algorithms, this tool computes routes that minimize total driving distance. The approach is suitable for trips with a large number of waypoints, where brute-force search is computationally infeasible.

## Features

- Easy customization of road trip waypoints.
- Automated querying of Google Maps for driving distances and durations between all pairs of waypoints.
- Efficient route optimization using a genetic algorithm.
- Outputs both the optimized route and a visualization-ready format for mapping tools.

## Requirements

- **Python 3.x**
- **pandas**: Data analysis and manipulation.
- **googlemaps**: Access to Google Maps Distance Matrix API.
- **numpy**: Numerical operations.

You can install the required packages using pip:

```bash
pip install pandas
pip install googlemaps
```

## Getting Started

### 1. Define Your Waypoints

Create a list of all the destinations you want to include in your road trip. Make sure each entry is a valid location as recognized by Google Maps. Example:

```python
all_waypoints = [
    "USS Alabama, Battleship Parkway, Mobile, AL",
    "Grand Canyon National Park, Arizona",
    "Toltec Mounds, Scott, AR",
    ...
    "Yellowstone National Park, WY 82190"
]
```

**Note:** The Google Maps API has a free tier with a limit of 70 waypoints per day. Exceeding this limit requires a paid plan.

### 2. Set Up Google Maps API

- Enable the **Google Maps Distance Matrix API** for your Google account.
- Obtain an API key and paste it in the code where specified.

```python
import googlemaps
gmaps = googlemaps.Client(key="YOUR_API_KEY_HERE")
```

### 3. Query Distances Between Waypoints

The script will automatically query Google Maps for the driving distance and duration between every pair of waypoints. Results are saved to a file to avoid redundant API requests.

### 4. Run the Genetic Algorithm

The genetic algorithm optimizes the order of waypoints to minimize total travel distance. Tweak parameters such as the number of generations and population size for your needs.

### 5. Visualize the Route

Once you have an optimized route, it can be visualized using mapping tools. The output is formatted for easy integration with third-party visualization libraries.

---

## License

This repository and its contents are provided under a liberal license to encourage wide use and sharing. Please see the LICENSE file for full details.

## Contact

If you have questions, suggestions, or issues, feel free to:

- Email: rajarshi.somvanshi@gmail.com
- Open an issue in this repository

I strive to respond within a couple of days.

---

## Technical Notes

- The genetic algorithm provides a high-quality solution but does not guarantee the absolute optimum. For mathematically optimal solutions, consider using analytical solvers such as Concorde TSP.
- The visualization of routes with more than 10 waypoints on Google Maps requires splitting the route due to API limitations. Refer to the comments in the code for guidance on visualization.

---

**Happy road tripping!**
