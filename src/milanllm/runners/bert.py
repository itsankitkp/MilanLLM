from milanllm.models.text import *
from milanllm.models.core import *

from milanllm.core.utils import *
from milanllm.models.text import _get_batch_loss_bert


def train():
    batch_size, max_len = 512, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)
    net = BERTModel(
        len(vocab),
        num_hiddens=128,
        ffn_num_hiddens=256,
        num_heads=2,
        num_blks=2,
        dropout=0.2,
    )
    devices = try_all_gpus()
    loss = nn.CrossEntropyLoss()
    train_bert(train_iter, net, loss, len(vocab), devices, 50)


def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):

    net(*next(iter(train_iter))[:4])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, Timer()
    animator = Animator(
        xlabel="step", ylabel="loss", xlim=[1, num_steps], legend=["mlm", "nsp"]
    )
    # Sum of masked language modeling losses, sum of next sentence prediction
    # losses, no. of sentence pairs, count
    metric = Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for (
            tokens_X,
            segments_X,
            valid_lens_x,
            pred_positions_X,
            mlm_weights_X,
            mlm_Y,
            nsp_y,
        ) in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net,
                loss,
                vocab_size,
                tokens_X,
                segments_X,
                valid_lens_x,
                pred_positions_X,
                mlm_weights_X,
                mlm_Y,
                nsp_y,
            )
        l.backward()
        print(f'loss: {l:.3f}')
        trainer.step()
        metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
        timer.stop()
        animator.add(step + 1, (metric[0] / metric[3], metric[1] / metric[3]))
        step += 1
        if step == num_steps:
            num_steps_reached = True
            break
    print(
        f"MLM loss {metric[0] / metric[3]:.3f}, "
        f"NSP loss {metric[1] / metric[3]:.3f}"
    )
    print(f"{metric[2] / timer.sum():.1f} sentence pairs/sec on " f"{str(devices)}")
