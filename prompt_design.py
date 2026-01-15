aux_prompt = '''You are a network traffic analysis expert. I will provide multiple time series samples representing HTTP user request behavior.
Each sample covers a 15-second window, with up to one record per second (i.e., 15 time steps). Only time steps with available data will be included—missing time steps indicate no recorded data.

Each time step contains the following fields in sequence:
1. The index of the time step, taking integer values from 0 to 14;
2. The total number of flows created during this time step.
3. The total number of packets transmitted in these flows;
4. The total number of bytes transmittedin these flows;
5. The average number of packets per flow;
6. The max number of packets per flow;
7. The minimum number of packets per flow;
8. The standard deviation of packets per flow;
9. The average bytes per flow;
10. The max bytes per flow;
11. The min bytes per flow;
12. The standard deviation of bytes per flow;
  
Your task is to determine if the following sample of behavior fits the following characteristics. Your determination will give insight into HTTP DDoS detection.

The characteristics to be judged are as follows:
1. High Request Rate. Normal users typically have a lower request rate, whereas attackers tend to generate requests at a higher rate. Since the number of flows created by a sample is positively correlated with the number of HTTP requests initiated, you can determine this by calculating the flow creation rate.
2. Continuous Requesting. Normal users typically request resources intermittently, while attackers tend to make continuous requests over multiple consecutive time step.
3. Fixed Behavior Pattern. Normal users generally have more diverse request patterns, while attackers tend to follow patterns with minimal variation. This should be assessed based on the statistical metrics within each time step. 

### Output Rules:
- Only output valid JSON in your reply.
- Do not include any explanatory text, markdown formatting, or extra line breaks.
- Stick exactly to the output format below.

### Output format:
{
  "sample_1": {
    "is high request rate": true/false,
    "is continuous requesting": true/false,
    "is fixed behavior pattern": true/false
  },
  "sample_2": {
    ......
}

Here are the samples:
'''

def get_aux_prompt():
    return aux_prompt