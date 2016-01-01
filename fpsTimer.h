#include <ctime>
#include <deque>

class FpsTimer
{
private:
	std::deque<float> *lastFrameTimes;
	time_t lastFrame, tempTime;
	char *fpsString;
	int averageOfFrames;
	int framesToUpdate;
	float averageFps;
public:
	FpsTimer(int averageOfFrames);
	void timeFrame();
	char *getFps();
};