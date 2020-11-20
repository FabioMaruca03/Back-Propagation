package back.propagation.neuron;

import java.util.* ;

public class N_units {

  private double value;
  private double in;
  private double delta;
  private final Map<Integer, Double> nW;
  private final int previousNodes;
  private final List<Double> ws;

  public N_units(int previous) {
    this.nW = new HashMap<>();
    this.ws = new ArrayList<>();
    for (int i = 0; i < previous; ++i) {
      ws.add(new Random().nextDouble() % 10 + 1);
    }

    this.in = 0;
    this.previousNodes = previous;
  }

  public void performActivateFunction() {
    value = sigmoid(in);
  }

  public void addInput(double in , int weightIndex) {
    this.in += in* ws.get(weightIndex);
  }

  public double sigmoid(double s) {
    return 1 / (1 + Math.pow(Math.E, -s));
  }

  public double sigmoidDerivative(double s) {
    return (1 - s) * s;
  }

  public double getValue() {
    return value;
  }

  public void setIn(double in) {
    this.in = in;
  }

  public int getNumberOfPreviousNodes() {
    return previousNodes;
  }

  public List<Double> getWs() {
    return ws;
  }

  public double getDelta() {
    return delta;
  }

  public void setDelta(double delta) {
    this.delta = delta;
  }

  public void setValue(double value) {
    this.value = value;
  }

  public void improveWeight(double gradient, int index) {
    double newWeight = this.getWs().get(index) - gradient;
    nW.put(index, newWeight);
  }

  public void perform() {
    if (nW != null) {
      for (Map.Entry<Integer, Double> entry: nW.entrySet()) {
        ws.set(entry.getKey(), entry.getValue());
      }
    }
  }

  @Override
  public String toString() {
    return "\nUnit{" + "value=" + value + '}';
  }
}