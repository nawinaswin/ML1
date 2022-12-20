#include "vector"
#include "iostream"
#include "cstdlib" 
#include "cassert"
#include "cmath"


 struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron;
typedef std::vector<Neuron> Layer;

//**************************************Class Neuron*******************************************

 class Neuron{
 public:
    Neuron(unsigned numOutputs);

    void setOutputVal(double val){m_outputVal = val;};
    double getOutputVal(void) const {return m_outputVal;}
    void feedForward(const Layer &prevLayer){};
    void calculateOutputGradients(double targetVal);
    void calculateHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
private:
    double m_outputVal;
    double eta;
    double alpha;
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void){return rand()/double(RAND_MAX);}
    double sumDOW(const Layer &nextLayer) const;
    std::vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

double Neuron::getResults()
{

}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;
    for(unsigned n = 0; n < nextLayer.size() - 1; ++n){
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
}
void Neuron::calculateHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
    for( unsigned n=0; n < prevLayer.size(); ++n)
    {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeights = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeights =  eta * neuron_getOutputVals * m_gradient + alpha * oldDeltaWeights;
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeights;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeights;
    }
}

double Neuron::transferFunction(double x)
{
    //tanh with output range of -1 to 1
    return(tanh(x))
    
}

void Neuron::calculateOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal); 
}

double Neuron::transferFunctionDerivative(double x)
{
    return(1 - x * x);
}

Neuron::Neuron(unsigned int numOutputs)
{
    for(unsigned c = 0; c < numOutputs; ++c)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
}

void Neuron::feedForward(const int &prevLayer)
{
    double sum=0.0;
    for (unsigned n = 0; n < prevLayer.size(); ++n)
    {
        sum += prevLayer[n].getOutputVal() *
                prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    m_outputVal = Neuron::transferFunction(sum);
}

//***************************************Class Net***********************************************

 class Net{
 public:
    Net(const std::vector<unsigned> &topology);
    void feedForward(const std::vector<double> &inputVals){};
    void backProp(const std::vector<double> &targetVals){};
    void getResults(const std::vector<double> &resultVals) const {};

 private:
    std::vector<Layer> m_layers;
    double m_error;
 };

Net::Net(const std::vector<unsigned> &topology)
{
    unsigned numLayer = topology.size();
    for(unsigned layerNum = 0; layerNum < numLayer; ++layerNum)
    {
        m_layers.push_back(Layer());
        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
        {
            unsigned numOutputs =  layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
            m_layers.back().push_back(Neuron(numOutputs));
            std::cout<<"Neuron added.\n";
        }
    }
}

void Net::feedForward(const std::vector<double> &inputVals)
{
    assert(inputVals.size() == m_layers.size()-1);

    //Assigning input vals to input neurons
    for(unsigned i = 0; i < inputVals.size(); ++i){
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    //forward propagate
    for(unsigned layerNum = 0; layerNum<m_layers.size(); ++layerNum)
    {
        Layer &prevLayer = m_layers[layerNum-1];
        for(unsigned n = 0; n < m_layers[layerNum].size(); ++n)
        {
        m_layers[layerNum][n].feedforward(prevLayer);
        
        }
    }
}

void Net::backProp(const std::vector<double> &targetVals)
{
    // calculate overall net error rms
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for(unsigned n = 0; n < outputLayer.size(); ++n)
    {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error);

    //recent average measure
    m_recentAverageError  =  (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)/
            (m_recentAverageSmoothingFactor + 1.0);

    //calculate output layer gradient
    for(unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        outputLayer[n].calculateOutputGradients(targetVals[n]);
    }

    //calculate hidden layer gradient
    for(unsigned layerNum = n_layer.size() - 2;layerNum > 0; --layerNum)
    {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for(unsigned n = 0; n < hiddenLayer; ++n)
        {
            hiddenLayer[n].calculateHiddenLayerGradients(nextLayer);
        }
    }

    //for all the layer from input to output and all hidden layers update connection weights
    for(unsigned layerNum=m_layers.size() - 1; layerNum > 0; --layerNum)
    { Layer &layer = m_layers[layerNum];
      Layer &prevLayer = m_layers[layerNum - 1];
      for(unsigned n = 0; n < layer.size(); ++n)
      {
          layer[n].updateInputWeights(prevLayer);
      }
    }


}
 int main() {

    std::vector<unsigned> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);

    Net myNet(topology);

    std::vector<double> inputVals;
    myNet.feedForward(inputVals);

    std::vector<double> targetVals;
    myNet.backProp(targetVals);

    std::vector<double> resultVals;
    myNet.getResults(resultVals);
    return 0;
}
