import json

classes = {
    "dog": ("🐕", "Dog barking", "Repeated dog barking", "Aggressive continuous barking"),
    "rooster": ("🐓", "Rooster crowing", "Multiple roosters", "Constant rooster noise"),
    "pig": ("🐖", "Pig oinking", "Loud pig squeals", "Distressed pig sounds"),
    "cow": ("🐄", "Cow mooing", "Herd of cows mooing", "Loud distressed mooing"),
    "frog": ("🐸", "Frog croaking", "Multiple frogs", "Loud chorus of frogs"),
    "cat": ("🐈", "Cat meowing", "Loud cat meowing", "Cat fighting/yowling"),
    "hen": ("🐔", "Hen clucking", "Loud hen noises", "Disturbed hens"),
    "insects": ("🦗", "Insects chirping", "Loud insect swarm", "Overwhelming insect noise"),
    "sheep": ("🐑", "Sheep baaing", "Flock of sheep", "Loud distressed sheep"),
    "crow": ("🐦‍⬛", "Crow cawing", "Multiple crows", "Flock of crows cawing loudly"),
    
    "rain": ("🌧️", "Light rain", "Heavy rain", "Torrential downpour"),
    "sea_waves": ("🌊", "Gentle waves", "Crashing waves", "Violent sea waves"),
    "crackling_fire": ("🔥", "Small crackling fire", "Large fire crackling", "Raging fire sounds"),
    "crickets": ("🦗", "Crickets chirping", "Loud crickets", "Deafening cricket noise"),
    "chirping_birds": ("🐦", "Birds chirping", "Loud bird calls", "Flock of noisy birds"),
    "water_drops": ("💧", "Water dripping", "Continuous dripping", "Heavy water leak"),
    "wind": ("🌬️", "Gentle breeze", "Strong winds", "Gale force winds/howling"),
    "pouring_water": ("🚰", "Water pouring", "Continuous pouring", "Water spilling/flooding"),
    "thunderstorm": ("⛈️", "Distant thunder", "Thunderstorm overhead", "Loud violent lightning strikes"),
    
    "crying_baby": ("👶", "Baby fussing", "Baby crying loudly", "Baby screaming continuously"),
    "sneezing": ("🤧", "Someone sneezing", "Repeated sneezing", "Continuous violent sneezing"),
    "clapping": ("👏", "Light clapping", "Applause", "Loud continuous applause"),
    "breathing": ("😮‍💨", "Heavy breathing", "Labored breathing", "Gasping for air"),
    "coughing": ("😷", "Someone coughing", "Repeated coughing fits", "Severe continuous coughing"),
    "footsteps": ("👣", "Footsteps nearby", "Heavy/rapid footsteps", "Someone running/fleeing"),
    "laughing": ("😂", "Light laughter", "Loud laughing", "Hysterical laughter"),
    "brushing_teeth": ("🪥", "Brushing teeth", "Vigorous brushing", "Loud electric toothbrush"),
    "snoring": ("😴", "Light snoring", "Heavy snoring", "Loud sleep apnea snoring"),
    "drinking_sipping": ("🥤", "Sipping a drink", "Loud drinking/gulping", "Choking while drinking"),
    
    "door_wood_knock": ("🚪", "Single knock on door", "Repeated knocking", "Frantic pounding on door"),
    "mouse_click": ("🖱️", "Mouse clicking", "Rapid clicking", "Frantic continuous clicking"),
    "keyboard_typing": ("⌨️", "Typing on keyboard", "Fast typing", "Aggressive continuous typing"),
    "door_wood_creaks": ("🚪", "Door creaking", "Loud door creak", "Multiple doors creaking open"),
    "can_opening": ("🥫", "Can opening", "Multiple cans opening", "Loud continuous popping"),
    "washing_machine": ("🧺", "Washing machine running", "Loud spin cycle", "Washing machine violently shaking"),
    "vacuum_cleaner": ("🧹", "Vacuum cleaner", "Loud vacuuming nearby", "Vacuum running continuously"),
    "clock_alarm": ("⏰", "Alarm beeping", "Loud alarm ringing", "Continuous ignored alarm"),
    "clock_tick": ("⏱️", "Clock ticking", "Loud ticking", "Frantic ticking/countdown"),
    "glass_breaking": ("💥", "Small glass clink", "Glass breaking", "Large window shattering violently"),
    "toilet_flush": ("🚽", "Toilet flushing", "Repeated flushing", "Toilet overflowing/running continuously"),
    
    "helicopter": ("🚁", "Helicopter in distance", "Helicopter nearby", "Helicopter directly overhead"),
    "chainsaw": ("🪚", "Chainsaw in distance", "Loud chainsaw operating", "Chainsaw dangerously close"),
    "siren": ("🚨", "Distant siren", "Approaching emergency vehicle", "Loud siren directly outside"),
    "car_horn": ("🚗", "Short car horn", "Repeated honking", "Continuous blaring car horn"),
    "engine": ("⚙️", "Engine idling", "Loud engine revving", "Engine backfiring/failing"),
    "train": ("🚂", "Distant train horn", "Train approaching", "Loud train passing by closely"),
    "church_bells": ("⛪", "Church bells ringing", "Loud church bells", "Continuous frantic bell ringing"),
    "airplane": ("✈️", "Airplane high above", "Low flying airplane", "Extremely loud low-altitude jet"),
    "fireworks": ("🎆", "Distant fireworks", "Loud fireworks display", "Explosive fireworks nearby"),
    "hand_saw": ("🪚", "Hand saw cutting", "Continuous sawing", "Aggressive fast sawing")
}

rules = {}
for k, (icon, green, yellow, red) in classes.items():
    rules[k] = {
        "icon": icon,
        "levels": {
            "green": {
                "situation": green,
                "action": "Acknowledge",
                "base_confidence": 0.5
            },
            "yellow": {
                "situation": yellow,
                "action": "Investigate",
                "base_confidence": 0.7
            },
            "red": {
                "situation": red,
                "action": "Immediate Action",
                "base_confidence": 0.9
            }
        }
    }

with open("rules.json", "w", encoding="utf-8") as f:
    json.dump(rules, f, indent=4)
print("done")
