Computing the optimal road trip across the U.S.
This notebook provides the methodology and code used in the blog post, Computing the optimal road trip across the U.S.

Notebook by Randal S. Olson
Please see the repository README file for the licenses and usage terms for the instructional material and code in this notebook. In general, I have licensed this material so that it is as widely useable and shareable as possible.

The code in this notebook is also available as a single Python script here courtesy of Andrew Liesinger.

Required Python libraries
If you don't have Python on your computer, you can use the Anaconda Python distribution to install most of the Python packages you need. Anaconda provides a simple double-click installer for your convenience.

This code uses base Python libraries except for googlemaps and pandas packages. You can install these packages using pip by typing the following commands into your command line:

pip install pandas

pip install googlemaps

Construct a list of road trip waypoints
The first step is to decide where you want to stop on your road trip.

Make sure you look all of the locations up on Google Maps first so you have the correct address, city, state, etc. If the text you use to look up the location doesn't work on Google Maps, then it won't work here either.

Add all of your waypoints to the list below. Make sure they're formatted the same way as in the example below.

Technical note: Due to daily usage limitations of the Google Maps API, you can only have a maximum of 70 waypoints. You will have to pay Google for an increased API limit if you want to add more waypoints.

all_waypoints = ["USS Alabama, Battleship Parkway, Mobile, AL",
                 "Grand Canyon National Park, Arizona",
                 "Toltec Mounds, Scott, AR",
                 "San Andreas Fault, San Benito County, CA",
                 "Cable Car Museum, 94108, 1201 Mason St, San Francisco, CA 94108",
                 "Pikes Peak, Colorado",
                 "The Mark Twain House & Museum, Farmington Avenue, Hartford, CT",
                 "New Castle Historic District, Delaware",
                 "White House, Pennsylvania Avenue Northwest, Washington, DC",
                 "Cape Canaveral, FL",
                 "Okefenokee Swamp Park, Okefenokee Swamp Park Road, Waycross, GA",
                 "Craters of the Moon National Monument & Preserve, Arco, ID",
                 "Lincoln Home National Historic Site Visitor Center, 426 South 7th Street, Springfield, IL",
                 "West Baden Springs Hotel, West Baden Avenue, West Baden Springs, IN",
                 "Terrace Hill, Grand Avenue, Des Moines, IA",
                 "C. W. Parker Carousel Museum, South Esplanade Street, Leavenworth, KS",
                 "Mammoth Cave National Park, Mammoth Cave Pkwy, Mammoth Cave, KY",
                 "French Quarter, New Orleans, LA",
                 "Acadia National Park, Maine",
                 "Maryland State House, 100 State Cir, Annapolis, MD 21401",
                 "USS Constitution, Boston, MA",
                 "Olympia Entertainment, Woodward Avenue, Detroit, MI",
                 "Fort Snelling, Tower Avenue, Saint Paul, MN",
                 "Vicksburg National Military Park, Clay Street, Vicksburg, MS",
                 "Gateway Arch, Washington Avenue, St Louis, MO",
                 "Glacier National Park, West Glacier, MT",
                 "Ashfall Fossil Bed, Royal, NE",
                 "Hoover Dam, NV",
                 "Omni Mount Washington Resort, Mount Washington Hotel Road, Bretton Woods, NH",
                 "Congress Hall, Congress Place, Cape May, NJ 08204",
                 "Carlsbad Caverns National Park, Carlsbad, NM",
                 "Statue of Liberty, Liberty Island, NYC, NY",
                 "Wright Brothers National Memorial Visitor Center, Manteo, NC",
                 "Fort Union Trading Post National Historic Site, Williston, North Dakota 1804, ND",
                 "Spring Grove Cemetery, Spring Grove Avenue, Cincinnati, OH",
                 "Chickasaw National Recreation Area, 1008 W 2nd St, Sulphur, OK 73086",
                 "Columbia River Gorge National Scenic Area, Oregon",
                 "Liberty Bell, 6th Street, Philadelphia, PA",
                 "The Breakers, Ochre Point Avenue, Newport, RI",
                 "Fort Sumter National Monument, Sullivan's Island, SC",
                 "Mount Rushmore National Memorial, South Dakota 244, Keystone, SD",
                 "Graceland, Elvis Presley Boulevard, Memphis, TN",
                 "The Alamo, Alamo Plaza, San Antonio, TX",
                 "Bryce Canyon National Park, Hwy 63, Bryce, UT",
                 "Shelburne Farms, Harbor Road, Shelburne, VT",
                 "Mount Vernon, Fairfax County, Virginia",
                 "Hanford Site, Benton County, WA",
                 "Lost World Caverns, Lewisburg, WV",
                 "Taliesin, County Road C, Spring Green, Wisconsin",
                 "Yellowstone National Park, WY 82190"]
Next you'll have to register this script with the Google Maps API so they know who's hitting their servers with hundreds of Google Maps routing requests.

Enable the Google Maps Distance Matrix API on your Google account. Google explains how to do that here.

Copy and paste the API key they had you create into the code below.

import googlemaps

gmaps = googlemaps.Client(key="PASTE YOUR API KEY HERE")
Now we're going to query the Google Maps API for the shortest route between all of the waypoints.

This is equivalent to doing Google Maps directions lookups on the Google Maps site, except now we're performing hundreds of lookups automatically using code.

If you get an error on this part, that most likely means one of the waypoints you entered couldn't be found on Google Maps. Another possible reason for an error here is if it's not possible to drive between the points, e.g., finding the driving directions between Hawaii and Florida will return an error until we invent flying cars.

Gather the distance traveled on the shortest route between all waypoints
from itertools import combinations

waypoint_distances = {}
waypoint_durations = {}

for (waypoint1, waypoint2) in combinations(all_waypoints, 2):
    try:
        route = gmaps.distance_matrix(origins=[waypoint1],
                                      destinations=[waypoint2],
                                      mode="driving", # Change this to "walking" for walking directions,
                                                      # "bicycling" for biking directions, etc.
                                      language="English",
                                      units="metric")

        # "distance" is in meters
        distance = route["rows"][0]["elements"][0]["distance"]["value"]

        # "duration" is in seconds
        duration = route["rows"][0]["elements"][0]["duration"]["value"]

        waypoint_distances[frozenset([waypoint1, waypoint2])] = distance
        waypoint_durations[frozenset([waypoint1, waypoint2])] = duration
    
    except Exception as e:
        print("Error with finding the route between %s and %s." % (waypoint1, waypoint2))
Now that we have the routes between all of our waypoints, let's save them to a text file so we don't have to bother Google about them again.

with open("my-waypoints-dist-dur.tsv", "w") as out_file:
    out_file.write("\t".join(["waypoint1",
                              "waypoint2",
                              "distance_m",
                              "duration_s"]))
    
    for (waypoint1, waypoint2) in waypoint_distances.keys():
        out_file.write("\n" +
                       "\t".join([waypoint1,
                                  waypoint2,
                                  str(waypoint_distances[frozenset([waypoint1, waypoint2])]),
                                  str(waypoint_durations[frozenset([waypoint1, waypoint2])])]))
Use a genetic algorithm to optimize the order to visit the waypoints in
Instead of exhaustively looking at every possible solution, genetic algorithms start with a handful of random solutions and continually tinkers with these solutions — always trying something slightly different from the current solutions and keeping the best ones — until they can’t find a better solution any more.

Below, all you need to do is make sure that the file name above matches the file name below (both currently my-waypoints-dist-dur.tsv) and run the code. The code will read in your route information and use a genetic algorithm to discover an optimized driving route.

import pandas as pd
import numpy as np

waypoint_distances = {}
waypoint_durations = {}
all_waypoints = set()

waypoint_data = pd.read_csv("my-waypoints-dist-dur.tsv", sep="\t")

for i, row in waypoint_data.iterrows():
    waypoint_distances[frozenset([row.waypoint1, row.waypoint2])] = row.distance_m
    waypoint_durations[frozenset([row.waypoint1, row.waypoint2])] = row.duration_s
    all_waypoints.update([row.waypoint1, row.waypoint2])
import random

def compute_fitness(solution):
    """
        This function returns the total distance traveled on the current road trip.
        
        The genetic algorithm will favor road trips that have shorter
        total distances traveled.
    """
    
    solution_fitness = 0.0
    
    for index in range(len(solution)):
        waypoint1 = solution[index - 1]
        waypoint2 = solution[index]
        solution_fitness += waypoint_distances[frozenset([waypoint1, waypoint2])]
        
    return solution_fitness

def generate_random_agent():
    """
        Creates a random road trip from the waypoints.
    """
    
    new_random_agent = list(all_waypoints)
    random.shuffle(new_random_agent)
    return tuple(new_random_agent)

def mutate_agent(agent_genome, max_mutations=3):
    """
        Applies 1 - `max_mutations` point mutations to the given road trip.
        
        A point mutation swaps the order of two waypoints in the road trip.
    """
    
    agent_genome = list(agent_genome)
    num_mutations = random.randint(1, max_mutations)
    
    for mutation in range(num_mutations):
        swap_index1 = random.randint(0, len(agent_genome) - 1)
        swap_index2 = swap_index1

        while swap_index1 == swap_index2:
            swap_index2 = random.randint(0, len(agent_genome) - 1)

        agent_genome[swap_index1], agent_genome[swap_index2] = agent_genome[swap_index2], agent_genome[swap_index1]
            
    return tuple(agent_genome)

def shuffle_mutation(agent_genome):
    """
        Applies a single shuffle mutation to the given road trip.
        
        A shuffle mutation takes a random sub-section of the road trip
        and moves it to another location in the road trip.
    """
    
    agent_genome = list(agent_genome)
    
    start_index = random.randint(0, len(agent_genome) - 1)
    length = random.randint(2, 20)
    
    genome_subset = agent_genome[start_index:start_index + length]
    agent_genome = agent_genome[:start_index] + agent_genome[start_index + length:]
    
    insert_index = random.randint(0, len(agent_genome) + len(genome_subset) - 1)
    agent_genome = agent_genome[:insert_index] + genome_subset + agent_genome[insert_index:]
    
    return tuple(agent_genome)

def generate_random_population(pop_size):
    """
        Generates a list with `pop_size` number of random road trips.
    """
    
    random_population = []
    for agent in range(pop_size):
        random_population.append(generate_random_agent())
    return random_population
    
def run_genetic_algorithm(generations=5000, population_size=100):
    """
        The core of the Genetic Algorithm.
        
        `generations` and `population_size` must be a multiple of 10.
    """
    
    population_subset_size = int(population_size / 10.)
    generations_10pct = int(generations / 10.)
    
    # Create a random population of `population_size` number of solutions.
    population = generate_random_population(population_size)

    # For `generations` number of repetitions...
    for generation in range(generations):
        
        # Compute the fitness of the entire current population
        population_fitness = {}

        for agent_genome in population:
            if agent_genome in population_fitness:
                continue

            population_fitness[agent_genome] = compute_fitness(agent_genome)

        # Take the top 10% shortest road trips and produce offspring each from them
        new_population = []
        for rank, agent_genome in enumerate(sorted(population_fitness,
                                                   key=population_fitness.get)[:population_subset_size]):
            
            if (generation % generations_10pct == 0 or generation == generations - 1) and rank == 0:
                print("Generation %d best: %d | Unique genomes: %d" % (generation,
                                                                       population_fitness[agent_genome],
                                                                       len(population_fitness)))
                print(agent_genome)
                print("")

            # Create 1 exact copy of each of the top road trips
            new_population.append(agent_genome)

            # Create 2 offspring with 1-3 point mutations
            for offspring in range(2):
                new_population.append(mutate_agent(agent_genome, 3))
                
            # Create 7 offspring with a single shuffle mutation
            for offspring in range(7):
                new_population.append(shuffle_mutation(agent_genome))

        # Replace the old population with the new population of offspring 
        for i in range(len(population))[::-1]:
            del population[i]

        population = new_population
Try running the genetic algorithm a few times to see the different solutions it comes up with. It should take about a minute to finish running.

run_genetic_algorithm(generations=5000, population_size=100)
Generation 0 best: 90432537 | Unique genomes: 100
('Mount Rushmore National Memorial, South Dakota 244, Keystone, SD', 'Olympia Entertainment, Woodward Avenue, Detroit, MI', 'Terrace Hill, Grand Avenue, Des Moines, IA', 'Statue of Liberty, Liberty Island, NYC, NY', 'Maryland State House, 100 State Cir, Annapolis, MD 21401', 'Shelburne Farms, Harbor Road, Shelburne, VT', 'Fort Union Trading Post National Historic Site, Williston, North Dakota 1804, ND', 'Gateway Arch, Washington Avenue, St Louis, MO', 'USS Alabama, Battleship Parkway, Mobile, AL', "Fort Sumter National Monument, Sullivan's Island, SC", 'Glacier National Park, West Glacier, MT', 'The Alamo, Alamo Plaza, San Antonio, TX', 'Hanford Site, Benton County, WA', 'Columbia River Gorge National Scenic Area, Oregon', 'White House, Pennsylvania Avenue Northwest, Washington, DC', 'Mammoth Cave National Park, Mammoth Cave Pkwy, Mammoth Cave, KY', 'Mount Vernon, Fairfax County, Virginia', 'Lost World Caverns, Lewisburg, WV', 'Graceland, Elvis Presley Boulevard, Memphis, TN', 'Taliesin, County Road C, Spring Green, Wisconsin', 'Wright Brothers National Memorial Visitor Center, Manteo, NC', 'The Breakers, Ochre Point Avenue, Newport, RI', 'Hoover Dam, NV', 'The Mark Twain House & Museum, Farmington Avenue, Hartford, CT', 'New Castle Historic District, Delaware', 'USS Constitution, Boston, MA', 'West Baden Springs Hotel, West Baden Avenue, West Baden Springs, IN', 'Fort Snelling, Tower Avenue, Saint Paul, MN', 'Chickasaw National Recreation Area, 1008 W 2nd St, Sulphur, OK 73086', 'Vicksburg National Military Park, Clay Street, Vicksburg, MS', 'Acadia National Park, Maine', 'Bryce Canyon National Park, Hwy 63, Bryce, UT', 'San Andreas Fault, San Benito County, CA', 'Ashfall Fossil Bed, Royal, NE', 'French Quarter, New Orleans, LA', 'C. W. Parker Carousel Museum, South Esplanade Street, Leavenworth, KS', 'Liberty Bell, 6th Street, Philadelphia, PA', 'Omni Mount Washington Resort, Mount Washington Hotel Road, Bretton Woods, NH', 'Carlsbad Caverns National Park, Carlsbad, NM', 'Pikes Peak, Colorado', 'Toltec Mounds, Scott, AR', 'Grand Canyon National Park, Arizona', 'Cable Car Museum, 94108, 1201 Mason St, San Francisco, CA 94108', 'Yellowstone National Park, WY 82190', 'Cape Canaveral, FL', 'Congress Hall, Congress Place, Cape May, NJ 08204', 'Okefenokee Swamp Park, Okefenokee Swamp Park Road, Waycross, GA', 'Spring Grove Cemetery, Spring Grove Avenue, Cincinnati, OH', 'Lincoln Home National Historic Site Visitor Center, 426 South 7th Street, Springfield, IL', 'Craters of the Moon National Monument & Preserve, Arco, ID')

Generation 1000 best: 22613362 | Unique genomes: 95
('Terrace Hill, Grand Avenue, Des Moines, IA', 'C. W. Parker Carousel Museum, South Esplanade Street, Leavenworth, KS', 'Ashfall Fossil Bed, Royal, NE', 'Fort Union Trading Post National Historic Site, Williston, North Dakota 1804, ND', 'Mount Rushmore National Memorial, South Dakota 244, Keystone, SD', 'Yellowstone National Park, WY 82190', 'Craters of the Moon National Monument & Preserve, Arco, ID', 'Glacier National Park, West Glacier, MT', 'Hanford Site, Benton County, WA', 'Columbia River Gorge National Scenic Area, Oregon', 'Cable Car Museum, 94108, 1201 Mason St, San Francisco, CA 94108', 'San Andreas Fault, San Benito County, CA', 'Hoover Dam, NV', 'Grand Canyon National Park, Arizona', 'Bryce Canyon National Park, Hwy 63, Bryce, UT', 'Pikes Peak, Colorado', 'Carlsbad Caverns National Park, Carlsbad, NM', 'The Alamo, Alamo Plaza, San Antonio, TX', 'Chickasaw National Recreation Area, 1008 W 2nd St, Sulphur, OK 73086', 'Toltec Mounds, Scott, AR', 'Graceland, Elvis Presley Boulevard, Memphis, TN', 'Vicksburg National Military Park, Clay Street, Vicksburg, MS', 'French Quarter, New Orleans, LA', 'USS Alabama, Battleship Parkway, Mobile, AL', 'Cape Canaveral, FL', 'Okefenokee Swamp Park, Okefenokee Swamp Park Road, Waycross, GA', "Fort Sumter National Monument, Sullivan's Island, SC", 'Wright Brothers National Memorial Visitor Center, Manteo, NC', 'Congress Hall, Congress Place, Cape May, NJ 08204', 'Statue of Liberty, Liberty Island, NYC, NY', 'Shelburne Farms, Harbor Road, Shelburne, VT', 'Omni Mount Washington Resort, Mount Washington Hotel Road, Bretton Woods, NH', 'Acadia National Park, Maine', 'USS Constitution, Boston, MA', 'The Breakers, Ochre Point Avenue, Newport, RI', 'The Mark Twain House & Museum, Farmington Avenue, Hartford, CT', 'Liberty Bell, 6th Street, Philadelphia, PA', 'New Castle Historic District, Delaware', 'Maryland State House, 100 State Cir, Annapolis, MD 21401', 'White House, Pennsylvania Avenue Northwest, Washington, DC', 'Mount Vernon, Fairfax County, Virginia', 'Lost World Caverns, Lewisburg, WV', 'Olympia Entertainment, Woodward Avenue, Detroit, MI', 'Spring Grove Cemetery, Spring Grove Avenue, Cincinnati, OH', 'Mammoth Cave National Park, Mammoth Cave Pkwy, Mammoth Cave, KY', 'West Baden Springs Hotel, West Baden Avenue, West Baden Springs, IN', 'Gateway Arch, Washington Avenue, St Louis, MO', 'Lincoln Home National Historic Site Visitor Center, 426 South 7th Street, Springfield, IL', 'Taliesin, County Road C, Spring Green, Wisconsin', 'Fort Snelling, Tower Avenue, Saint Paul, MN')

Generation 2000 best: 22324940 | Unique genomes: 98
('Graceland, Elvis Presley Boulevard, Memphis, TN', 'Vicksburg National Military Park, Clay Street, Vicksburg, MS', 'French Quarter, New Orleans, LA', 'USS Alabama, Battleship Parkway, Mobile, AL', 'Cape Canaveral, FL', 'Okefenokee Swamp Park, Okefenokee Swamp Park Road, Waycross, GA', "Fort Sumter National Monument, Sullivan's Island, SC", 'Wright Brothers National Memorial Visitor Center, Manteo, NC', 'Congress Hall, Congress Place, Cape May, NJ 08204', 'Shelburne Farms, Harbor Road, Shelburne, VT', 'Omni Mount Washington Resort, Mount Washington Hotel Road, Bretton Woods, NH', 'Acadia National Park, Maine', 'USS Constitution, Boston, MA', 'The Breakers, Ochre Point Avenue, Newport, RI', 'The Mark Twain House & Museum, Farmington Avenue, Hartford, CT', 'Statue of Liberty, Liberty Island, NYC, NY', 'Liberty Bell, 6th Street, Philadelphia, PA', 'New Castle Historic District, Delaware', 'Maryland State House, 100 State Cir, Annapolis, MD 21401', 'White House, Pennsylvania Avenue Northwest, Washington, DC', 'Mount Vernon, Fairfax County, Virginia', 'Lost World Caverns, Lewisburg, WV', 'Olympia Entertainment, Woodward Avenue, Detroit, MI', 'Spring Grove Cemetery, Spring Grove Avenue, Cincinnati, OH', 'Mammoth Cave National Park, Mammoth Cave Pkwy, Mammoth Cave, KY', 'West Baden Springs Hotel, West Baden Avenue, West Baden Springs, IN', 'Gateway Arch, Washington Avenue, St Louis, MO', 'Lincoln Home National Historic Site Visitor Center, 426 South 7th Street, Springfield, IL', 'Taliesin, County Road C, Spring Green, Wisconsin', 'Fort Snelling, Tower Avenue, Saint Paul, MN', 'Terrace Hill, Grand Avenue, Des Moines, IA', 'C. W. Parker Carousel Museum, South Esplanade Street, Leavenworth, KS', 'Ashfall Fossil Bed, Royal, NE', 'Mount Rushmore National Memorial, South Dakota 244, Keystone, SD', 'Fort Union Trading Post National Historic Site, Williston, North Dakota 1804, ND', 'Glacier National Park, West Glacier, MT', 'Yellowstone National Park, WY 82190', 'Craters of the Moon National Monument & Preserve, Arco, ID', 'Hanford Site, Benton County, WA', 'Columbia River Gorge National Scenic Area, Oregon', 'Cable Car Museum, 94108, 1201 Mason St, San Francisco, CA 94108', 'San Andreas Fault, San Benito County, CA', 'Hoover Dam, NV', 'Grand Canyon National Park, Arizona', 'Bryce Canyon National Park, Hwy 63, Bryce, UT', 'Pikes Peak, Colorado', 'Carlsbad Caverns National Park, Carlsbad, NM', 'The Alamo, Alamo Plaza, San Antonio, TX', 'Chickasaw National Recreation Area, 1008 W 2nd St, Sulphur, OK 73086', 'Toltec Mounds, Scott, AR')

Generation 3000 best: 22324940 | Unique genomes: 95
('Graceland, Elvis Presley Boulevard, Memphis, TN', 'Vicksburg National Military Park, Clay Street, Vicksburg, MS', 'French Quarter, New Orleans, LA', 'USS Alabama, Battleship Parkway, Mobile, AL', 'Cape Canaveral, FL', 'Okefenokee Swamp Park, Okefenokee Swamp Park Road, Waycross, GA', "Fort Sumter National Monument, Sullivan's Island, SC", 'Wright Brothers National Memorial Visitor Center, Manteo, NC', 'Congress Hall, Congress Place, Cape May, NJ 08204', 'Shelburne Farms, Harbor Road, Shelburne, VT', 'Omni Mount Washington Resort, Mount Washington Hotel Road, Bretton Woods, NH', 'Acadia National Park, Maine', 'USS Constitution, Boston, MA', 'The Breakers, Ochre Point Avenue, Newport, RI', 'The Mark Twain House & Museum, Farmington Avenue, Hartford, CT', 'Statue of Liberty, Liberty Island, NYC, NY', 'Liberty Bell, 6th Street, Philadelphia, PA', 'New Castle Historic District, Delaware', 'Maryland State House, 100 State Cir, Annapolis, MD 21401', 'White House, Pennsylvania Avenue Northwest, Washington, DC', 'Mount Vernon, Fairfax County, Virginia', 'Lost World Caverns, Lewisburg, WV', 'Olympia Entertainment, Woodward Avenue, Detroit, MI', 'Spring Grove Cemetery, Spring Grove Avenue, Cincinnati, OH', 'Mammoth Cave National Park, Mammoth Cave Pkwy, Mammoth Cave, KY', 'West Baden Springs Hotel, West Baden Avenue, West Baden Springs, IN', 'Gateway Arch, Washington Avenue, St Louis, MO', 'Lincoln Home National Historic Site Visitor Center, 426 South 7th Street, Springfield, IL', 'Taliesin, County Road C, Spring Green, Wisconsin', 'Fort Snelling, Tower Avenue, Saint Paul, MN', 'Terrace Hill, Grand Avenue, Des Moines, IA', 'C. W. Parker Carousel Museum, South Esplanade Street, Leavenworth, KS', 'Ashfall Fossil Bed, Royal, NE', 'Mount Rushmore National Memorial, South Dakota 244, Keystone, SD', 'Fort Union Trading Post National Historic Site, Williston, North Dakota 1804, ND', 'Glacier National Park, West Glacier, MT', 'Yellowstone National Park, WY 82190', 'Craters of the Moon National Monument & Preserve, Arco, ID', 'Hanford Site, Benton County, WA', 'Columbia River Gorge National Scenic Area, Oregon', 'Cable Car Museum, 94108, 1201 Mason St, San Francisco, CA 94108', 'San Andreas Fault, San Benito County, CA', 'Hoover Dam, NV', 'Grand Canyon National Park, Arizona', 'Bryce Canyon National Park, Hwy 63, Bryce, UT', 'Pikes Peak, Colorado', 'Carlsbad Caverns National Park, Carlsbad, NM', 'The Alamo, Alamo Plaza, San Antonio, TX', 'Chickasaw National Recreation Area, 1008 W 2nd St, Sulphur, OK 73086', 'Toltec Mounds, Scott, AR')

Generation 4000 best: 22324940 | Unique genomes: 97
('Graceland, Elvis Presley Boulevard, Memphis, TN', 'Vicksburg National Military Park, Clay Street, Vicksburg, MS', 'French Quarter, New Orleans, LA', 'USS Alabama, Battleship Parkway, Mobile, AL', 'Cape Canaveral, FL', 'Okefenokee Swamp Park, Okefenokee Swamp Park Road, Waycross, GA', "Fort Sumter National Monument, Sullivan's Island, SC", 'Wright Brothers National Memorial Visitor Center, Manteo, NC', 'Congress Hall, Congress Place, Cape May, NJ 08204', 'Shelburne Farms, Harbor Road, Shelburne, VT', 'Omni Mount Washington Resort, Mount Washington Hotel Road, Bretton Woods, NH', 'Acadia National Park, Maine', 'USS Constitution, Boston, MA', 'The Breakers, Ochre Point Avenue, Newport, RI', 'The Mark Twain House & Museum, Farmington Avenue, Hartford, CT', 'Statue of Liberty, Liberty Island, NYC, NY', 'Liberty Bell, 6th Street, Philadelphia, PA', 'New Castle Historic District, Delaware', 'Maryland State House, 100 State Cir, Annapolis, MD 21401', 'White House, Pennsylvania Avenue Northwest, Washington, DC', 'Mount Vernon, Fairfax County, Virginia', 'Lost World Caverns, Lewisburg, WV', 'Olympia Entertainment, Woodward Avenue, Detroit, MI', 'Spring Grove Cemetery, Spring Grove Avenue, Cincinnati, OH', 'Mammoth Cave National Park, Mammoth Cave Pkwy, Mammoth Cave, KY', 'West Baden Springs Hotel, West Baden Avenue, West Baden Springs, IN', 'Gateway Arch, Washington Avenue, St Louis, MO', 'Lincoln Home National Historic Site Visitor Center, 426 South 7th Street, Springfield, IL', 'Taliesin, County Road C, Spring Green, Wisconsin', 'Fort Snelling, Tower Avenue, Saint Paul, MN', 'Terrace Hill, Grand Avenue, Des Moines, IA', 'C. W. Parker Carousel Museum, South Esplanade Street, Leavenworth, KS', 'Ashfall Fossil Bed, Royal, NE', 'Mount Rushmore National Memorial, South Dakota 244, Keystone, SD', 'Fort Union Trading Post National Historic Site, Williston, North Dakota 1804, ND', 'Glacier National Park, West Glacier, MT', 'Yellowstone National Park, WY 82190', 'Craters of the Moon National Monument & Preserve, Arco, ID', 'Hanford Site, Benton County, WA', 'Columbia River Gorge National Scenic Area, Oregon', 'Cable Car Museum, 94108, 1201 Mason St, San Francisco, CA 94108', 'San Andreas Fault, San Benito County, CA', 'Hoover Dam, NV', 'Grand Canyon National Park, Arizona', 'Bryce Canyon National Park, Hwy 63, Bryce, UT', 'Pikes Peak, Colorado', 'Carlsbad Caverns National Park, Carlsbad, NM', 'The Alamo, Alamo Plaza, San Antonio, TX', 'Chickasaw National Recreation Area, 1008 W 2nd St, Sulphur, OK 73086', 'Toltec Mounds, Scott, AR')

Generation 4999 best: 22324940 | Unique genomes: 99
('Graceland, Elvis Presley Boulevard, Memphis, TN', 'Vicksburg National Military Park, Clay Street, Vicksburg, MS', 'French Quarter, New Orleans, LA', 'USS Alabama, Battleship Parkway, Mobile, AL', 'Cape Canaveral, FL', 'Okefenokee Swamp Park, Okefenokee Swamp Park Road, Waycross, GA', "Fort Sumter National Monument, Sullivan's Island, SC", 'Wright Brothers National Memorial Visitor Center, Manteo, NC', 'Congress Hall, Congress Place, Cape May, NJ 08204', 'Shelburne Farms, Harbor Road, Shelburne, VT', 'Omni Mount Washington Resort, Mount Washington Hotel Road, Bretton Woods, NH', 'Acadia National Park, Maine', 'USS Constitution, Boston, MA', 'The Breakers, Ochre Point Avenue, Newport, RI', 'The Mark Twain House & Museum, Farmington Avenue, Hartford, CT', 'Statue of Liberty, Liberty Island, NYC, NY', 'Liberty Bell, 6th Street, Philadelphia, PA', 'New Castle Historic District, Delaware', 'Maryland State House, 100 State Cir, Annapolis, MD 21401', 'White House, Pennsylvania Avenue Northwest, Washington, DC', 'Mount Vernon, Fairfax County, Virginia', 'Lost World Caverns, Lewisburg, WV', 'Olympia Entertainment, Woodward Avenue, Detroit, MI', 'Spring Grove Cemetery, Spring Grove Avenue, Cincinnati, OH', 'Mammoth Cave National Park, Mammoth Cave Pkwy, Mammoth Cave, KY', 'West Baden Springs Hotel, West Baden Avenue, West Baden Springs, IN', 'Gateway Arch, Washington Avenue, St Louis, MO', 'Lincoln Home National Historic Site Visitor Center, 426 South 7th Street, Springfield, IL', 'Taliesin, County Road C, Spring Green, Wisconsin', 'Fort Snelling, Tower Avenue, Saint Paul, MN', 'Terrace Hill, Grand Avenue, Des Moines, IA', 'C. W. Parker Carousel Museum, South Esplanade Street, Leavenworth, KS', 'Ashfall Fossil Bed, Royal, NE', 'Mount Rushmore National Memorial, South Dakota 244, Keystone, SD', 'Fort Union Trading Post National Historic Site, Williston, North Dakota 1804, ND', 'Glacier National Park, West Glacier, MT', 'Yellowstone National Park, WY 82190', 'Craters of the Moon National Monument & Preserve, Arco, ID', 'Hanford Site, Benton County, WA', 'Columbia River Gorge National Scenic Area, Oregon', 'Cable Car Museum, 94108, 1201 Mason St, San Francisco, CA 94108', 'San Andreas Fault, San Benito County, CA', 'Hoover Dam, NV', 'Grand Canyon National Park, Arizona', 'Bryce Canyon National Park, Hwy 63, Bryce, UT', 'Pikes Peak, Colorado', 'Carlsbad Caverns National Park, Carlsbad, NM', 'The Alamo, Alamo Plaza, San Antonio, TX', 'Chickasaw National Recreation Area, 1008 W 2nd St, Sulphur, OK 73086', 'Toltec Mounds, Scott, AR')

Visualize your road trip on a Google map
Now that we have an ordered list of the waypoints, we should put them on a Google map so we can see the trip from a high level and make any extra adjustments.

There's no easy way to make this visualization in Python, but the Google Maps team provides a nice JavaScript library for visualizing routes on a Google Map.

Here's an example map with the route between 50 waypoints visualized: link

The tricky part here is that the JavaScript library only plots routes with a maximum of 10 waypoints. If we want to plot a route with >10 waypoints, we need to call the route plotting function multiple times.

Thanks to some optimizations by Nicholas Clarke to my original map, this is a simple operation:

Copy the final route generated by the genetic algorithm above.

Place brackets ([ & ]) around the route, e.g.,

['Graceland, Elvis Presley Boulevard, Memphis, TN', 'Vicksburg National Military Park, Clay Street, Vicksburg, MS', 'French Quarter, New Orleans, LA', 'USS Alabama, Battleship Parkway, Mobile, AL', 'Cape Canaveral, FL', 'Okefenokee Swamp Park, Okefenokee Swamp Park Road, Waycross, GA', "Fort Sumter National Monument, Sullivan's Island, SC", 'Wright Brothers National Memorial Visitor Center, Manteo, NC', 'Congress Hall, Congress Place, Cape May, NJ 08204', 'Shelburne Farms, Harbor Road, Shelburne, VT', 'Omni Mount Washington Resort, Mount Washington Hotel Road, Bretton Woods, NH', 'Acadia National Park, Maine', 'USS Constitution, Boston, MA', 'The Breakers, Ochre Point Avenue, Newport, RI', 'The Mark Twain House & Museum, Farmington Avenue, Hartford, CT', 'Statue of Liberty, Liberty Island, NYC, NY', 'Liberty Bell, 6th Street, Philadelphia, PA', 'New Castle Historic District, Delaware', 'Maryland State House, 100 State Cir, Annapolis, MD 21401', 'White House, Pennsylvania Avenue Northwest, Washington, DC', 'Mount Vernon, Fairfax County, Virginia', 'Lost World Caverns, Lewisburg, WV', 'Olympia Entertainment, Woodward Avenue, Detroit, MI', 'Spring Grove Cemetery, Spring Grove Avenue, Cincinnati, OH', 'Mammoth Cave National Park, Mammoth Cave Pkwy, Mammoth Cave, KY', 'West Baden Springs Hotel, West Baden Avenue, West Baden Springs, IN', 'Gateway Arch, Washington Avenue, St Louis, MO', 'Lincoln Home National Historic Site Visitor Center, 426 South 7th Street, Springfield, IL', 'Taliesin, County Road C, Spring Green, Wisconsin', 'Fort Snelling, Tower Avenue, Saint Paul, MN', 'Terrace Hill, Grand Avenue, Des Moines, IA', 'C. W. Parker Carousel Museum, South Esplanade Street, Leavenworth, KS', 'Ashfall Fossil Bed, Royal, NE', 'Mount Rushmore National Memorial, South Dakota 244, Keystone, SD', 'Fort Union Trading Post National Historic Site, Williston, North Dakota 1804, ND', 'Glacier National Park, West Glacier, MT', 'Yellowstone National Park, WY 82190', 'Craters of the Moon National Monument & Preserve, Arco, ID', 'Hanford Site, Benton County, WA', 'Columbia River Gorge National Scenic Area, Oregon', 'Cable Car Museum, 94108, 1201 Mason St, San Francisco, CA 94108', 'San Andreas Fault, San Benito County, CA', 'Hoover Dam, NV', 'Grand Canyon National Park, Arizona', 'Bryce Canyon National Park, Hwy 63, Bryce, UT', 'Pikes Peak, Colorado', 'Carlsbad Caverns National Park, Carlsbad, NM', 'The Alamo, Alamo Plaza, San Antonio, TX', 'Chickasaw National Recreation Area, 1008 W 2nd St, Sulphur, OK 73086', 'Toltec Mounds, Scott, AR']

Paste the final route with brackets into line 93 of my road trip map code. It should look like this:

optimal_route = [ ... ]

where ... is your optimized road trip.

That's all it takes! Now you have your own optimized road trip ready to show off to the world.

Some technical notes
As I mentioned in the original article, by the end of 5,000 generations, the genetic algorithm will very likely find a good but probably not the absolute best solution to the optimal routing problem. It is in the nature of genetic algorithms that we never know if we found the absolute best solution.

However, there exist some brilliant analytical solutions to the optimal routing problem such as the Concorde TSP solver. If you're interested in learning more about Concorde and how it's possible to find a perfect solution to the routing problem, I advise you check out Bill Cook's article on the topic.

If you have any questions
Please feel free to:

Email me,

Tweet at me, or

comment on the blog post

I'm usually pretty good about getting back to people within a day or two.
