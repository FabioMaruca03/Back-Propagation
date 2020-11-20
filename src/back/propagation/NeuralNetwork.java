package back.propagation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import back.propagation.data.Data;
import back.propagation.layers.H_layers;
import back.propagation.layers.N_layers;
import back.propagation.neuron.H_units;
import back.propagation.neuron.O_Units;

public class NeuralNetwork {

  private final double LEARNING_RATE = 0.5;

  private final Data data;

  private int nINs;
  private final int nHL;
  private final int[] hLS;

  private final N_layers n_layers;
  private final List<H_layers> h_layers;
  private final O_Units o_Units;

  private double err;

  private int i;

  public NeuralNetwork(Data d, int nHL, int[] hLS) {
    this.data = d;
    this.nHL = Math.max(nHL, 2);
    this.hLS = hLS;

    if (!data.getX().isEmpty() && !data.getX().get(0).isEmpty()) {
      this.nINs = data.getX().get(0).size();
      this.i = 0;
    }

    this.n_layers = new N_layers();
    initializeInputLayer();

    this.h_layers = new ArrayList < >();
    initializeHiddenLayers();

    this.o_Units = new O_Units(h_layers.get(nHL - 1).getNumberOfUnits());

  }

  public void clearInputs() {
    for (int i = 0; i < nHL; ++i) {
      for (int j = 0; j < h_layers.get(i).getNumberOfUnits(); ++j) {
        h_layers.get(i).getHiddenUnits().get(j).setIn(0.0);
      }
    }

    o_Units.setIn(0.0);
  }

  public void test(List<List<Double>> realX, List < Double > realY) {

    System.out.println(this.toString());

    normalizeTestData(realX, realY);

    double totalError = 0;

    for (int i = 0; i < realX.size(); ++i) {
      forwardTest(realX, realY, i);

      totalError += Math.abs(o_Units.geteV() - o_Units.getValue());

      clearInputs();

      double realValue = realY.get(i) * (data.getMaxY() - data.getMinY()) + data.getMinY();
      double valuePredicted = o_Units.getValue() * (data.getMaxY() - data.getMinY()) + data.getMinY();
      System.out.println("Real value: " + realValue + " | Predicted value: " + valuePredicted);
    }

    System.out.println("Error rate: " + totalError / realY.size() * 100);

  }

  private void normalizeTestData(List<List<Double>> realX, List<Double> realY) {
    for (List<Double> x : realX) {
      for (int j = 0; j < x.size(); ++j) {
        double normalizedValue = (x.get(j) - data.getMinX().get(j)) / (data.getMaxX().get(j) - data.getMinX().get(j));
        x.set(j, normalizedValue);
      }
    }

    for (int i = 0; i < realX.size(); ++i) {
      double normalizedValue = (realY.get(i) - data.getMinY()) / (data.getMaxY() - data.getMinY());
      realY.set(i, normalizedValue);
    }
  }

  public void forwardTest(List < List < Double >> realX, List < Double > realY, int i) {

    o_Units.seteV(realY.get(i));

    for (int j = 0; j < nINs; ++j) {
      n_layers.getInputNodes().get(j).setValue(realX.get(i).get(j));
    }

    forwardInputToFirstHiddenLayer();
    forwardHiddenLayers();
    forwardLastHiddenLayerToOutputUnit();

  }

  public void train(boolean printResults, boolean showErrorRate) {

    double totalError = 0;

    while (i < data.getY().size()) {
      forward();

      if (printResults) {
        printResult();
      }

      backPropagation();
      calculateError();
      totalError += Math.abs(o_Units.geteV() - o_Units.getValue());

      clearInputs();

      i++;
    }

    if (showErrorRate) {
      System.out.println("Error rate: " + totalError / data.getY().size() * 100);
    }
    i = 0;

  }

  public void backPropagation() {

    calculateError();
    backPropagateOutputWeights();
    backPropagateHiddenLayersWeights();
    performImprovements();

  }

  public void performImprovements() {
    o_Units.perform();
    for (int i = 0; i < nHL; ++i) {
      for (int j = 0; j < h_layers.get(i).getNumberOfUnits(); ++j) {
        h_layers.get(i).getHiddenUnits().get(j).perform();
      }
    }
  }

  private void backPropagateHiddenLayersWeights() {

    for (int i = 0; i < h_layers.get(nHL - 1).getNumberOfUnits(); ++i) {
      H_units theHiddenUnit = h_layers.get(nHL - 1).getHiddenUnits().get(i);

      double gradient = o_Units.getDelta() * o_Units.getWs().get(i);
      gradient *= (theHiddenUnit.getValue() * (1 - theHiddenUnit.getValue()));
      theHiddenUnit.setDelta(gradient);
      for (int j = 0; j < h_layers.get(nHL - 2).getNumberOfUnits(); ++j) {
        double inputValue = h_layers.get(nHL - 2).getHiddenUnits().get(j).getValue();
        theHiddenUnit.improveWeight(gradient * inputValue * this.LEARNING_RATE, j);
      }
    }

    for (int i = nHL - 2; i > 0; --i) {
      backPropagateHiddenLayer(i);
    }

    for (int i = 0; i < h_layers.get(0).getNumberOfUnits(); ++i) {
      H_units theHiddenUnit = h_layers.get(0).getHiddenUnits().get(i);

      double gradient = 0.0;
      for (int j = 0; j < h_layers.get(1).getNumberOfUnits(); ++j) {
        gradient += h_layers.get(1).getHiddenUnits().get(j).getDelta() * h_layers.get(1).getHiddenUnits().get(j).getWs().get(i);
      }

      gradient *= (theHiddenUnit.getValue() * (1 - theHiddenUnit.getValue()));
      theHiddenUnit.setDelta(gradient);
      for (int j = 0; j < n_layers.getInputNodes().size(); ++j) {
        double inputValue = n_layers.getInputNodes().get(j).getValue();
        theHiddenUnit.improveWeight(gradient * inputValue * this.LEARNING_RATE, j);
      }
    }

  }

  private void backPropagateHiddenLayer(int aux) {

    for (int i = 0; i < h_layers.get(aux).getNumberOfUnits(); ++i) {
      H_units theHiddenUnit = h_layers.get(aux).getHiddenUnits().get(i);

      double gradient = 0.0;
      for (int j = 0; j < h_layers.get(aux + 1).getNumberOfUnits(); ++j) {
        gradient += h_layers.get(aux + 1).getHiddenUnits().get(j).getDelta() * h_layers.get(aux + 1).getHiddenUnits().get(j).getWs().get(i);
      }

      gradient *= (theHiddenUnit.getValue() * (1 - theHiddenUnit.getValue()));
      theHiddenUnit.setDelta(gradient);
      for (int j = 0; j < h_layers.get(aux - 1).getNumberOfUnits(); ++j) {
        double inputValue = h_layers.get(aux - 1).getHiddenUnits().get(j).getValue();
        theHiddenUnit.improveWeight(gradient * inputValue * this.LEARNING_RATE, j);
      }
    }

  }

  private void backPropagateOutputWeights() {

    for (int i = 0; i < o_Units.getNumberOfPreviousNodes(); ++i) {
      double gradient = -(o_Units.geteV() - o_Units.getValue());
      gradient *= ((o_Units.getValue() * (1 - o_Units.getValue())));
      o_Units.setDelta(gradient);
      gradient *= h_layers.get(nHL - 1).getHiddenUnits().get(i).getValue();

      o_Units.improveWeight(gradient * this.LEARNING_RATE, i);

    }
  }

  public void printResult() {
    System.out.println("--- Data: " + i + " -----------------------------------------------");
    System.out.println("Expected: " + data.getY().get(i) * (data.getMaxY() - data.getMinY()) + data.getMinY());
    System.out.println("Prediction: " + o_Units.getValue() * (data.getMaxY() - data.getMinY()) + data.getMinY());
    calculateError();
    System.out.println("Error: " + err);
    System.out.println("-------------------------------------------------------------\n");
  }

  public void calculateError() {
    this.err = Math.pow((data.getY().get(i) - o_Units.getValue()), 2) / 2;
  }

  public void forward() {

    o_Units.seteV(data.getY().get(i));

    for (int i = 0; i < nINs; ++i) {
      n_layers.getInputNodes().get(i).setValue(data.getX().get(this.i).get(i));
    }

    forwardInputToFirstHiddenLayer();
    forwardHiddenLayers();
    forwardLastHiddenLayerToOutputUnit();

  }

  private void forwardLastHiddenLayerToOutputUnit() {
    for (int i = 0; i < h_layers.get(nHL - 1).getNumberOfUnits(); ++i) {
      o_Units.addInput(h_layers.get(nHL - 1).getHiddenUnits().get(i).getValue(), i);
    }

    o_Units.performActivateFunction();
  }

  private void forwardHiddenLayers() {
    for (int i = 1; i < nHL; ++i) {
      for (int j = 0; j < h_layers.get(i).getNumberOfUnits(); ++j) {
        for (int k = 0; k < h_layers.get(i - 1).getNumberOfUnits(); ++k) {
          h_layers.get(i).getHiddenUnits().get(j).addInput(h_layers.get(i - 1).getHiddenUnits().get(k).getValue(), k);
        }
      }

      for (int j = 0; j < h_layers.get(i).getNumberOfUnits(); ++j) {
        h_layers.get(i).getHiddenUnits().get(j).performActivateFunction();
      }
    }
  }

  private void forwardInputToFirstHiddenLayer() {
    for (int i = 0; i < nINs; ++i) {
      for (int j = 0; j < h_layers.get(0).getNumberOfUnits(); ++j) {
        h_layers.get(0).getHiddenUnits().get(j).addInput(n_layers.getInputNodes().get(i).getValue(), i);
      }
    }

    for (int i = 0; i < h_layers.get(0).getNumberOfUnits(); ++i) {
      h_layers.get(0).getHiddenUnits().get(i).performActivateFunction();
    }
  }

  private void initializeInputLayer() {
    for (int i = 0; i < nINs; ++i) {
      n_layers.addInputNode(new Node());
    }
  }

  private void initializeHiddenLayers() {
    for (int i = 0; i < nHL; ++i) {
      int aux;

      if (i < hLS.length) {
        h_layers.add(new H_layers(hLS[i]));
        aux = hLS[i];
      } else {
        int DEFAULT = 3;
        h_layers.add(new H_layers(DEFAULT));
        aux = DEFAULT;
      }

      for (int j = 0; j < aux; ++j) {
        if (i == 0) {
          h_layers.get(i).addHiddenUnit(new H_units(nINs));
        } else {
          h_layers.get(i).addHiddenUnit(new H_units(h_layers.get(i - 1).getNumberOfUnits()));
        }
      }
    }
  }

  @Override
  public String toString() {
    return "Network{" + "LEARNING_RATE=" + LEARNING_RATE + ", nInputNodes=" + nINs + ", nHiddenLayers=" + nHL + ", hiddenLayersSizes=" + Arrays.toString(hLS) + '}';
  }
}