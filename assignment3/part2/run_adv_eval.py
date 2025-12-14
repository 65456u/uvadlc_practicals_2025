import argparse
import torch

from cifar10_models import resnet18
from utils import load_cifar10, train, test
from adversarial_attack import test_attack
from globals import STANDARD, FGSM, PGD, ALPHA, EPSILON, NUM_ITER


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build strategy args
    strategy_args = {
        STANDARD: None,
        FGSM: {ALPHA: args.alpha_fgsm, EPSILON: args.epsilon_fgsm},
        PGD: {ALPHA: args.alpha_pgd, EPSILON: args.epsilon_pgd, NUM_ITER: args.num_iter_pgd},
    }

    # Data
    trainloader, validloader, testloader, _ = load_cifar10(
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        augmentations=args.augmentations,
    )

    # Model
    model = resnet18(pretrained=args.pretrained).to(device)

    # Optional training/fine-tuning
    if args.num_epochs > 0:
        print(f"Training with defense={args.defense}")
        model = train(
            model,
            trainloader,
            validloader,
            num_epochs=args.num_epochs,
            defense_strategy=args.defense,
            defense_args=strategy_args[args.defense],
        )
    else:
        print("Skipping training (using pretrained weights)")

    # Clean accuracy
    clean_acc = test(model, testloader)
    print(f"Clean test accuracy: {clean_acc:.4f}")

    # FGSM attack eval
    fgsm_acc, fgsm_examples = test_attack(model, testloader, FGSM, strategy_args[FGSM])
    print(f"FGSM attack accuracy: {fgsm_acc:.4f}")

    # PGD attack eval
    pgd_acc, pgd_examples = test_attack(model, testloader, PGD, strategy_args[PGD])
    print(f"PGD attack accuracy: {pgd_acc:.4f}")

    if args.visualise and fgsm_examples:
        print("Saved adversarial examples during test_attack (see adversarial_examples directory)")

    if args.out_file:
        lines = [
            f"pretrained={args.pretrained}",
            f"defense={args.defense}",
            f"augmentations={args.augmentations}",
            f"epochs={args.num_epochs}",
            f"batch_size={args.batch_size}",
            f"epsilon_fgsm={args.epsilon_fgsm}",
            f"alpha_fgsm={args.alpha_fgsm}",
            f"epsilon_pgd={args.epsilon_pgd}",
            f"alpha_pgd={args.alpha_pgd}",
            f"num_iter_pgd={args.num_iter_pgd}",
            f"clean_acc={clean_acc:.4f}",
            f"fgsm_acc={fgsm_acc:.4f}",
            f"pgd_acc={pgd_acc:.4f}",
            ""
        ]
        with open(args.out_file, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate clean / FGSM / PGD accuracy")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--valid_ratio", type=float, default=0.75)
    parser.add_argument("--augmentations", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=0, help=">0 to fine-tune")
    parser.add_argument("--defense", choices=[STANDARD, FGSM, PGD], default=STANDARD)
    parser.add_argument("--epsilon_fgsm", type=float, default=0.1)
    parser.add_argument("--alpha_fgsm", type=float, default=0.5)
    parser.add_argument("--epsilon_pgd", type=float, default=0.03)
    parser.add_argument("--alpha_pgd", type=float, default=0.007)
    parser.add_argument("--num_iter_pgd", type=int, default=10)
    parser.add_argument("--visualise", action="store_true")
    parser.add_argument("--out_file", type=str, default="",
                        help="If set, append results and config to this file")
    args = parser.parse_args()
    main(args)
