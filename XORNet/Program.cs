using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;


//テストデータ作成
var trainData = new float[,]
{
    { 0, 0 },
    { 1, 0 },
    { 0, 1 },
    { 1, 1 },
};
//ラベル
var trainLabel = new float[,]
{
    {0},{1},{1},{0},
};
var model = new Net();
model.train();
var x = tensor(trainData);
var y = tensor(trainLabel);

const int EPOCH = 10000;
const double LR = 0.1;

//train
var optimizer = optim.SGD(model.parameters(), LR);
var sx = model.parameters();
for (var ep = 1; ep <= EPOCH; ++ep) {
    var eval = model.Forward(x);
    var loss = mse_loss(eval, y);
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    if (ep % 100 == 0)
        Console.WriteLine($"Epoch:{ep} Loss:{loss.ToSingle()}");
}
//評価１
model.eval();

model.Forward(torch.tensor(new float[] { 0, 0 })).print();

model.Forward(torch.tensor(new float[] { 0, 1 })).print();

model.Forward(torch.tensor(new float[] { 1, 0 })).print();

model.Forward(torch.tensor(new float[] { 1, 1 })).print();


//モデルのsave,load,評価2
model.save("test.bin");
var model2 = new Net();

model2.load("test.bin");
model2.eval();

model2.Forward(torch.tensor(new float[] { 0, 0 })).print();

model2.Forward(torch.tensor(new float[] { 0, 1 })).print();

model2.Forward(torch.tensor(new float[] { 1, 0 })).print();

model2.Forward(torch.tensor(new float[] { 1, 1 })).print();

class Net : nn.Module {
    public Net() : base("test") {
        this.linear1 = Linear(2, 2);
        this.linear2 = Linear(2, 1);
        RegisterComponents();
    }

    public Tensor Forward(Tensor x) {
        x = linear1.forward(x);
        x = functional.Sigmoid(x);
        return linear2.forward(x);
    }

    private Module<Tensor, Tensor> linear1;
    private Module<Tensor, Tensor> linear2;


}