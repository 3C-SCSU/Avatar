# NAO6 Gangnam Style Dance

This project allows the NAO6 robot to perform a simple version of "Gangnam Style" upon request.

## Behavior
- Responds to "Do, Gangnam Style" with "All right!"
- Plays the "Gangnam Style" music.
- Dances for ~30 seconds (simple hopping left and right).
- Stops music and movement after 30 seconds.
- And says "Hope you enjoyed it!"

## How to Run

1. Upload `gangnam_style.mp3` to `/home/nao/music/` on your NAO6 robot.
2. Update the `ROBOT_IP` variable in `robot_controller.py` with your NAO6's IP address.
3. Deploy and run the script:

```bash
python gangnam_choreograph.py
```