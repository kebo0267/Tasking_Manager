#Generate Class to read TLE data from file and save it in json format to file
from datetime import datetime, timedelta, timezone
import json
from operator import index
import os
import random
from TLE import TLE 
from pathlib import Path
from matplotlib import pyplot as plt


class TLEProcessor:
    def __init__(self, tle_file, json_file=None):
        self.tle_file = tle_file
        self.json_file = json_file
        self.data_directory = None
        self.tle_data = []
        self.tle_start_time: datetime = datetime.now(timezone.utc)
        self.tle_duration_hours: int = 2
        self.tle_duration_minutes: int = 0
        self.tle_end_time: datetime = self.tle_start_time + timedelta(hours=self.tle_duration_hours, minutes=self.tle_duration_minutes)

    def set_tle_start_time(self, start_time: datetime) -> None:
        self.tle_start_time = start_time
        self.tle_end_time = self.tle_start_time + timedelta(hours=self.tle_duration_hours, minutes=self.tle_duration_minutes)
        
    def set_tle_duration(self, hours: int, minutes: int=0) -> None:
        self.tle_duration_hours = hours
        self.tle_duration_minutes = minutes
        self.tle_end_time = self.tle_start_time + timedelta(hours=self.tle_duration_hours, minutes=self.tle_duration_minutes)   

    def set_tle_end_time(self, end_time: datetime) -> None:
        self.tle_end_time = end_time
        self.tle_duration_hours = int((self.tle_end_time - self.tle_start_time).total_seconds() // 3600)
        self.tle_duration_minutes = int(((self.tle_end_time - self.tle_start_time).total_seconds() % 3600) // 60)

    def set_data_directory(self, data_directory: str) -> None:
        self.data_directory = data_directory
        
    def read_tle_data(self) -> None:
        # Implementation for reading TLE data from file
        
        with open(self.tle_file, 'r') as f:
            while True:
                # Read the three lines of TLE data
                sat_name = f.readline().strip()
                if not sat_name:  # End of file
                    break
                tle_line1 = f.readline().strip()
                tle_line2 = f.readline().strip()
                # Pass the three lines to a TLE Object for processing
                tle_object = TLE()
                tle_object.parse_tle_from_data(sat_name, tle_line1, tle_line2)
                self.tle_data.append(tle_object)
        print(f"Total TLE entries read: {len(self.tle_data)}")

    def save_to_json(self) -> None:
        # Implementation for saving data to JSON file
        if not self.json_file is None:
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump([tle.get_data_as_dict() for tle in self.tle_data], f,indent=4 )

    # Read TLE data from JSON file and create TLE objects
    def read_from_json(self) -> None:
        
        with open(self.tle_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Total TLE entries read from JSON file: {len(data)}")
            for tle_dict in data:
                tle_object = TLE()
                tle_object.parse_tle_from_dict(tle_dict)
                self.tle_data.append(tle_object)
        print(f"Total TLE entries read from JSON: {len(self.tle_data)}")

    # Main method to process TLE data
    def process(self) -> None:
        print (f"Processing TLE data from file: {self.tle_file}")
        if self.tle_file.endswith('.json'):
            print (f"Reading TLE data from JSON file: {self.tle_file}")
            self.read_from_json()
        else:
            print (f"Reading TLE data from text file: {self.tle_file}")
            self.read_tle_data()
            self.save_to_json()


    # Metod to retrieve TLE data as a list filtered by satellite name
    def get_tle_data_by_name(self, satellite_name: str=None) -> list:
        if satellite_name is None:
            return [tle for tle in self.tle_data]
        else:
            return [tle for tle in self.tle_data if tle.get_satellite_name() == satellite_name]    
        
    # Metod to retrieve TLE data as a list by index or index range
    def get_tle_data_by_index(self, start_index: int=0, end_index: int=None) -> list:
        if start_index < len(self.tle_data) and end_index is None:
            return [ self.tle_data[start_index] ]
        elif end_index is not None and start_index < end_index  and end_index <= len(self.tle_data):
            return [tle for tle in self.tle_data[start_index:end_index]]
        
        
    def plot_ground_tracks(self, satellite_names: list=None) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'hspace': 0.3})
        for sat_name in satellite_names:
            get_tle_data = self.get_tle_data_by_name(sat_name)
            
            for tle in get_tle_data:
                # Plot ground track, latitude, longitude and altitude for each TLE object in one figure with 4 subplots
                # increase spacing between subplots
                tle.plot_ground_track(axes[0, 0])
                tle.plot_Altitude(axes[0, 1])
                tle.plot_Latitude(axes[1, 0])
                tle.plot_Longitude(axes[1, 1])

        handle, label = [], []
        for ax in axes.flatten():
            h, l = ax.get_legend_handles_labels()
            handle.extend(h)
            label.extend(l)
        
        unique_labels = dict(zip(label, handle))
        fig.legend(handles=unique_labels.values(), labels=unique_labels.keys(), loc='upper right', fontsize='small')
        plt.show()
            
    def plot_distance_between_satellites(self, satellite_name1: str, satellite_name2: str, in_view_only: bool=False) -> None:
        tle_data1 = self.get_tle_data_by_name(satellite_name1)
        tle_data2 = self.get_tle_data_by_name(satellite_name2)
        
        if not tle_data1 or not tle_data2:
            print(f"One or both satellites not found: {satellite_name1}, {satellite_name2}")
            return
        
        tle1 = tle_data1[0]
        tle2 = tle_data2[0]
        tle1.plot_distance_to_other_satellite(tle2, in_view_only=in_view_only)

    # Create histogram of satellite inclination angles
    def plot_inclination_histogram(self,bins: int=200) -> None:
        inclinations = [tle.get_inclination() for tle in self.tle_data]
        plt.hist(inclinations, bins=bins, edgecolor='black')
        plt.title('Histogram of Satellite Inclination Angles')
        plt.xlabel('Inclination (degrees)')
        plt.ylabel('Number of Satellites')
        plt.grid(axis='y', alpha=0.75)
        plt.show()

    # Create plot of satellite footprint radius vs time for a given satellite
    def plot_footprint_radius_over_time(self, satellite_name: str) -> None:
        tle_data = self.get_tle_data_by_name(satellite_name)
        
        if not tle_data:
            print(f"Satellite not found: {satellite_name}")
            return
        
        tle = tle_data[0]
        #times = tle.times
        footprint_radius_km = tle.get_footprint_radius_km()
        
        plt.plot(footprint_radius_km)
        plt.title(f'Footprint Radius Over Time for {satellite_name}')
        plt.xlabel('Time (UTC)')
        plt.ylabel('Footprint Radius (km)')
        plt.xticks(rotation=45)
        plt.grid()
        plt.tight_layout()
        plt.show()

   # Create Method to pick a set number of random satellites based on inclination angle and save their TLE data to a new JSON file
    def save_random_satellites_by_inclination(self, num_satellites: int, inclination_range: tuple, output_json_file: str, append: bool) -> None:

        filtered_tle_data = [tle for tle in self.tle_data if inclination_range[0] <= tle.get_inclination() <= inclination_range[1]]
        
        if len(filtered_tle_data) < num_satellites:
            print(f"Not enough satellites found in the specified inclination range. Found {len(filtered_tle_data)} satellites.")
            return
        
        random_satellites = [] 
        counter = 0
        while len(random_satellites) < num_satellites:
            flip = random.randint(0, 5)
            if flip == 5:
                random_satellites.append(filtered_tle_data[counter])
            counter += 1
            if counter >= len(filtered_tle_data):
                counter = 0
        
        
        if append:
            with open(output_json_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            existing_data.extend([tle.get_data_as_dict() for tle in random_satellites])
            with open(output_json_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=4)
        else:
            with open(output_json_file, 'w', encoding='utf-8') as f:
                json.dump([tle.get_data_as_dict() for tle in random_satellites], f, indent=4)

# Compare fov between all satellites.
    def compare_fov_between_satellites(self) -> None:
        index = 1
        for tle in self.tle_data:
            print(f"Comparing FOV for satellite: {tle.get_satellite_name()}")
            for other_tle in self.tle_data[index:]:
                print(f"Comparing {tle.get_satellite_name()} with satellite: {other_tle.get_satellite_name()}")
                tle.fov_overlaps_with_other_satellite(other_tle)
            index += 1
            self.save_individual_satellite_data_to_json(tle.get_satellite_info())

    # Create method to save data to JSON file with the name of file is the satellite name and the data is stored in
    # /data/training_data/{satellite_name}.json
    def save_individual_satellite_data_to_json(self, satellite: dict) -> None:
        output_dir = Path('data/training_data')
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        satellite_name = satellite["sat_name"]
        
        output_json_file = f"data/training_data/{satellite_name}.json"
        with open(output_json_file, 'w', encoding='utf-8') as f:
            print(f"Saving data for satellite: {satellite_name} to file: {output_json_file}")
            json.dump(satellite, f)
            
    # Load file from data/training_data/{satellite_name}.json and comare the fov data with the original TLE data for the satellite and print the results
    def compare_fov_with_saved_data(self, satellite_name_1: str, satellite_name_2: str) -> None:
        tle_data_1 = self.get_tle_data_by_name(satellite_name_1)[0]
        tle_data_2 = self.get_tle_data_by_name(satellite_name_2)[0] 

        if not tle_data_1 or not tle_data_2:
            print(f"One or both satellites not found: {satellite_name_1}, {satellite_name_2}")
            return

        if satellite_name_2 in tle_data_1.fov_intercepts.keys():
            print(f'Satellite {satellite_name_2} is in {satellite_name_1}\'s FOV')
        if satellite_name_1 in tle_data_2.fov_intercepts.keys():
            print(f'Satellite {satellite_name_1} is in {satellite_name_2}\'s FOV')
        print(f'{satellite_name_1} FOV Intercepts contains {len(tle_data_1.fov_intercepts[satellite_name_2])} entries')
        print(f'{satellite_name_2} FOV Intercepts contains {len(tle_data_2.fov_intercepts[satellite_name_1])} entries')

    # Create method to load data from JSON files in a given directory and saves them into self.tle_data as TLE objects
    def load_data_from_json_directory(self, directory: str) -> None:
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                file_path = os.path.join(directory, filename)
                tle_object = TLE()
                tle_object.load_data_from_json_file(file_path)
                self.tle_data.append(tle_object)
        print(f"Total TLE entries loaded from JSON directory: {len(self.tle_data)}")

if __name__ == "__main__":
    print(Path.cwd())
    
    # Check if data directory exists and move working directory up one level if it does not to find the data directory
    if not Path('data').exists():
        print("Data directory not found in current working directory. Moving up one level to find data directory.")
        os.chdir('..')
    print(Path.cwd())
    #tle_processor = TLEProcessor("data/starlink.txt", "data/starlink.json")
    tle_processor = TLEProcessor('data/training_data_starlink.json')
    
    # Set TLE start time to current UTC time and duration to 2 hours
    tle_processor.set_tle_start_time(datetime.now(timezone.utc) - timedelta(minutes=60))
    tle_processor.set_tle_duration(hours=26, minutes=0)
    tle_processor.set_data_directory('data/demo_data')
    # Set global TLE start time and duration in TLE class
    TLE.TLE_START_TIME = tle_processor.tle_start_time
    TLE.TLE_END_TIME = tle_processor.tle_end_time
    TLE.TLE_DEFAULT_FOV_ANGLE_DEG = 20
    tle_processor.load_data_from_json_directory('data/demo_data')

    #tle_processor.process()
    

    #tle_processor.plot_ground_tracks(["STARLINK-4467","STARLINK-34277"])
    #tle_processor.plot_distance_between_satellites("STARLINK-4467", "STARLINK-34277", in_view_only=True)
    #tle_processor.plot_inclination_histogram(100)
    #tle_processor.plot_footprint_radius_over_time("STARLINK-34277")
    #tle_processor.save_random_satellites_by_inclination(num_satellites=50, inclination_range=(0, 50), output_json_file='data/training_data_starlink.json', append=False)
    #tle_processor.save_random_satellites_by_inclination(num_satellites=50, inclination_range=(50, 60), output_json_file='data/training_data_starlink.json', append=True)
    #tle_processor.save_random_satellites_by_inclination(num_satellites=50, inclination_range=(60, 75), output_json_file='data/training_data_starlink.json', append=True)
    #tle_processor.save_random_satellites_by_inclination(num_satellites=50, inclination_range=(85, 100), output_json_file='data/training_data_starlink.json', append=True)
    tle_processor.compare_fov_between_satellites()