> output.txt
echo "LMS" >> output.txt
echo "" >> output.txt
echo "Batch Gradient Descent" >> output.txt
python batchGradientDescent.py >> output.txt
echo "" >> output.txt
echo "Stochastic Gradient Descent" >> output.txt
python stochasticGradientDescent.py >> output.txt
echo "" >> output.txt
echo "------------------------------" >> output.txt
echo "" >> output.txt
echo "Perceptrons" >> output.txt
echo "" >> output.txt
echo "Standard Perceptron" >> output.txt
python perceptron.py >> output.txt
echo "" >> output.txt
echo "Voted Perceptron" >> output.txt
python votedPerceptron.py >> output.txt
echo "" >> output.txt
echo "Average Perceptron" >> output.txt
python averagePerceptron.py >> output.txt
