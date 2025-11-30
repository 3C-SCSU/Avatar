Testing done by Kevin Gutierrez and Yash Patel on 11/26/25

1. We started by callibrating the headset using OpenBCI until every sensor was in the green range of resistance. 
2. We started the GUI5 project (python3 GUI5.py)
3. We selected Deep Learning
4. We verified that the model was using Live Data, and that the PyTorch button was selected
5. The headset wearer was shown a flight command for 3 seconds and was instructed to think about it. Then the Read my mind... button was clicked by the tester at the computer.
6. The log log was printed in the console log (in the GUI)

Items 4 through 6 were repeated for more brainwave reading attempts


Results: Automatic sending of the commands to the drone works as expected. We could not get live predictions using live data, pytorch, and deep learning (all together) to reliably predict the movement
