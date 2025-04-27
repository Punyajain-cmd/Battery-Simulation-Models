class BatteryManagementSystem:
    def __init__(self, capacity, voltage):
        self.capacity = capacity          # mAh
        self.voltage = voltage            # V
        self.soc = 100                    # Initial SoC (%)
        self.soh = 100                    # Initial SoH (%)
        self.voltage_profile = []         # Voltage data
    
    def update_soc(self, current, time):
        # Simple Coulomb counting
        charge = current * time / 3600  # mAh
        self.soc -= (charge / self.capacity) * 100
        self.soc = max(0, min(100, self.soc))
    
    def update_soh(self, cycle_count):
        # Simple degradation model: capacity fade with cycles
        degradation_rate = 0.05  # 5% degradation per 1000 cycles
        self.soh = max(0, 100 - (cycle_count / 1000) * degradation_rate * 100)
    
    def fault_detection(self, current, voltage):
        if voltage < 3.0:  # Low voltage fault
            return "Low Voltage Fault"
        if current > 10:  # Overcurrent fault
            return "Overcurrent Fault"
        return "No Fault"

# Simulation
bms = BatteryManagementSystem(3000, 3.7)  # 3000mAh, 3.7V
for cycle in range(100):
    bms.update_soc(1, 3600)  # Discharge at 1A for 1 hour
    bms.update_soh(cycle)
    print(f"Cycle {cycle}, SoC: {bms.soc:.2f}%, SoH: {bms.soh:.2f}%, Fault: {bms.fault_detection(1, 3.7)}")
