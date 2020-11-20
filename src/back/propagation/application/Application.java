package back.propagation.application;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.*;

import back.propagation.NeuralNetwork;
import back.propagation.data.Data;

public class Application {

  public static void main(String[] args) {

    Scanner s = new Scanner(System.in);
    List <List<Double>> x = new LinkedList<>();
    List <Double> y = new LinkedList<>();

    System.out.print("Type the Train ID (0 until 5): ");
    int ID = Integer.parseInt(s.nextLine())%5;
    String train = "Data/A1-turbine-train"+((ID==0)?"":ID)+".txt";
    System.out.println("Train data from turbine: " + train);
    readFile(x, y, train);

    System.out.print("Type int the HIDDEN LAYER SIZE (format: n, n, ...): ");
    int[] hiddenSize = Arrays.stream(s.nextLine().split(", ")).mapToInt(Integer::parseInt).toArray();

    NeuralNetwork neuralNetwork = new NeuralNetwork(new Data(x, y), 2, hiddenSize);

    long start = System.currentTimeMillis();


    System.out.print("Type in the MAX_EPOCH: ");
    final int MAX_EPOCH = s.nextInt();

    System.out.println("Max epochs: " + MAX_EPOCH);

    for (int epochs = 0; epochs < MAX_EPOCH; ++epochs) {
      neuralNetwork.train(false, true);
    }

    System.out.println("Execution time: " + (System.currentTimeMillis() - start) / 1000.0);

    x.clear(); y.clear();

    String test = "data/A1-turbine-test"+((ID==0)?"":ID)+".txt";
    System.out.println("Test data: " + test);
    readFile(x, y, test);

    neuralNetwork.test(x, y);

  }

  private static void readFile(List<List<Double>> incomingX, List<Double> incomingY, String path) {
    try {
      BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));

      String temp;

      while ((temp = br.readLine()) != null) {
        String[] slices = temp.split(" ");
        List<Double> doubles = new ArrayList <>();

        for (int i = 0; i < slices.length; ++i) {
          if (i == slices.length - 1) {
            incomingY.add(Double.parseDouble(slices[i]));
          } else {
            doubles.add(Double.parseDouble(slices[i]));
          }
        }

        incomingX.add(doubles);
      }
      br.close();
    } catch(Exception e) {
      e.printStackTrace();
    }
  }
}