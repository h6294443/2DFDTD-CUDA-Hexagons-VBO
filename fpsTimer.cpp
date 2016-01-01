#include "FpsTimer.h"

FpsTimer::FpsTimer(int averageOfFrames) {
	lastFrame = NULL;
	this->averageOfFrames = averageOfFrames;
	lastFrameTimes = new std::deque<float>(averageOfFrames);
	framesToUpdate = averageOfFrames;
	fpsString = new char[15];
}

void FpsTimer::timeFrame() {
	tempTime = clock();

	if (lastFrame != NULL) {
		if (lastFrameTimes->size() >= averageOfFrames) {
			lastFrameTimes->pop_back();
		}
		lastFrameTimes->push_front(tempTime - lastFrame);
	}
	lastFrame = tempTime;
}


char *FpsTimer::getFps(){
	framesToUpdate--;
	if (lastFrameTimes->size() < averageOfFrames) {
		return "Calculating";
	}

	if (framesToUpdate <= 0) {
		averageFps = 0;
		for (int i = 0; i < lastFrameTimes->size(); i++) {
			averageFps += lastFrameTimes->at(i);
		}
		averageFps /= lastFrameTimes->size();
		averageFps = CLOCKS_PER_SEC / averageFps;
		sprintf(fpsString, "%4.2f FPS", averageFps);
		framesToUpdate = averageOfFrames;
	}
	return fpsString;
}