#include "MLPComponent.h"
#include "Engine/TextureRenderTarget2D.h"
#include "TextureResource.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/PlayerController.h"
#include "Engine/World.h"
#include "DrawDebugHelpers.h"
#include "TimerManager.h"

UMLPComponent::UMLPComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

void UMLPComponent::BeginPlay()
{
	Super::BeginPlay();
	InitializeNetwork();
}

void UMLPComponent::InitializeNetwork()
{
	InputActivations.SetNumZeroed(InputNeuronCount);
	HiddenActivations.SetNumZeroed(HiddenNeuronCount);
	OutputActivations.SetNumZeroed(OutputNeuronCount);

	WeightsInputHidden.SetNum(InputNeuronCount * HiddenNeuronCount);
	HiddenBiases.SetNum(HiddenNeuronCount);
	WeightsHiddenOutput.SetNum(HiddenNeuronCount * OutputNeuronCount);
	OutputBiases.SetNum(OutputNeuronCount);

	OutputDeltas.SetNumZeroed(OutputNeuronCount);
	HiddenDeltas.SetNumZeroed(HiddenNeuronCount);

	// Poids faibles pour éviter la saturation de la sigmoïde sur 1024 entrées
	for (float& W : WeightsInputHidden) { W = FMath::FRandRange(-0.02f, 0.02f); }
	for (float& B : HiddenBiases) { B = 0.0f; }
	for (float& W : WeightsHiddenOutput) { W = FMath::FRandRange(-0.05f, 0.05f); }
	for (float& B : OutputBiases) { B = 0.0f; }
}

float UMLPComponent::Sigmoid(float X) const
{
	const float ClampedX = FMath::Clamp(X, -50.0f, 50.0f);
	return 1.0f / (1.0f + FMath::Exp(-ClampedX));
}

float UMLPComponent::SigmoidDerivative(float SigmoidOutput) const
{
	return SigmoidOutput * (1.0f - SigmoidOutput);
}

void UMLPComponent::ForwardPass()
{
	for (int32 h = 0; h < HiddenNeuronCount; ++h)
	{
		float WeightedSum = HiddenBiases[h];
		const int32 RowOffset = h * InputNeuronCount;

		for (int32 i = 0; i < InputNeuronCount; ++i)
		{
			WeightedSum += InputActivations[i] * WeightsInputHidden[RowOffset + i];
		}

		HiddenActivations[h] = Sigmoid(WeightedSum);
	}

	for (int32 o = 0; o < OutputNeuronCount; ++o)
	{
		float WeightedSum = OutputBiases[o];
		const int32 RowOffset = o * HiddenNeuronCount;

		for (int32 h = 0; h < HiddenNeuronCount; ++h)
		{
			WeightedSum += HiddenActivations[h] * WeightsHiddenOutput[RowOffset + h];
		}

		OutputActivations[o] = Sigmoid(WeightedSum);
	}
}

void UMLPComponent::Backpropagate(float ExpectedX, float ExpectedY)
{
	const float Expected[OutputNeuronCount] = { ExpectedX, ExpectedY };

	for (int32 o = 0; o < OutputNeuronCount; ++o)
	{
		const float dCost_dOutput = 2.0f * (OutputActivations[o] - Expected[o]);
		const float dOutput_dWI = SigmoidDerivative(OutputActivations[o]);
		OutputDeltas[o] = dCost_dOutput * dOutput_dWI;
	}

	for (int32 h = 0; h < HiddenNeuronCount; ++h)
	{
		float ErrorSum = 0.0f;
		for (int32 o = 0; o < OutputNeuronCount; ++o)
		{
			ErrorSum += OutputDeltas[o] * WeightsHiddenOutput[o * HiddenNeuronCount + h];
		}

		HiddenDeltas[h] = ErrorSum * SigmoidDerivative(HiddenActivations[h]);
	}

	for (int32 o = 0; o < OutputNeuronCount; ++o)
	{
		const int32 RowOffset = o * HiddenNeuronCount;
		for (int32 h = 0; h < HiddenNeuronCount; ++h)
		{
			WeightsHiddenOutput[RowOffset + h] -= LearningRate * OutputDeltas[o] * HiddenActivations[h];
		}
		OutputBiases[o] -= LearningRate * OutputDeltas[o];
	}

	for (int32 h = 0; h < HiddenNeuronCount; ++h)
	{
		const int32 RowOffset = h * InputNeuronCount;
		for (int32 i = 0; i < InputNeuronCount; ++i)
		{
			WeightsInputHidden[RowOffset + i] -= LearningRate * HiddenDeltas[h] * InputActivations[i];
		}
		HiddenBiases[h] -= LearningRate * HiddenDeltas[h];
	}
}

bool UMLPComponent::ExtractRedInputsAndTarget(float& OutExpectedX, float& OutExpectedY)
{
	if (!VisionRenderTarget)
	{
		UE_LOG(LogTemp, Error, TEXT("[MLP] Erreur : VisionRenderTarget est NULL !"));
		return false;
	}

	FTextureRenderTargetResource* RTResource = VisionRenderTarget->GameThread_GetRenderTargetResource();
	if (!RTResource)
	{
		UE_LOG(LogTemp, Error, TEXT("[MLP] Erreur : RTResource est NULL !"));
		return false;
	}

	TArray<FColor> Pixels;
	if (!RTResource->ReadPixels(Pixels) || Pixels.Num() < InputNeuronCount)
	{
		UE_LOG(LogTemp, Error, TEXT("[MLP] Erreur : ReadPixels a echoue (Pixels=%d)."), Pixels.Num());
		return false;
	}


	float MaxRedActivation = 0.0f;
	float BestRawRed = -999.0f;
	FColor SamplePixel = FColor::Black;

	for (int32 i = 0; i < InputNeuronCount; ++i)
	{
		const float R = Pixels[i].R / 255.0f;
		const float G = Pixels[i].G / 255.0f;
		const float B = Pixels[i].B / 255.0f;

		const float RedOverGreen = R - G;
		const float RedOverBlue = R - B;
		const float Yellowish = FMath::Max(0.0f, G - B);

		// Tolère l'éclairage chaud tout en rejetant l'orange
		const float RawRed = FMath::Min(RedOverGreen, RedOverBlue) - Yellowish;
		const float Activation = FMath::Clamp(RawRed * 4.0f, 0.0f, 1.0f);

		InputActivations[i] = Activation;

		if (RawRed > BestRawRed)
		{
			BestRawRed = RawRed;
			MaxRedActivation = Activation;
			SamplePixel = Pixels[i];
		}
	}

	TArray<bool> Visited;
	Visited.SetNumZeroed(InputNeuronCount);

	float BestClusterWeight = 0.0f;
	float BestCentroidX = 0.5f;
	float BestCentroidY = 0.5f;

	for (int32 StartIdx = 0; StartIdx < InputNeuronCount; ++StartIdx)
	{
		if (Visited[StartIdx] || InputActivations[StartIdx] < RedThreshold)
		{
			continue;
		}

		TArray<int32> Queue;
		Queue.Add(StartIdx);
		Visited[StartIdx] = true;

		float ClusterWeight = 0.0f;
		float SumX = 0.0f;
		float SumY = 0.0f;

		int32 Head = 0;
		while (Head < Queue.Num())
		{
			const int32 Curr = Queue[Head++];
			const int32 X = Curr % GridSize;
			const int32 Y = Curr / GridSize;
			const float W = InputActivations[Curr];

			ClusterWeight += W;
			SumX += ((X + 0.5f) / static_cast<float>(GridSize)) * W;
			SumY += ((Y + 0.5f) / static_cast<float>(GridSize)) * W;

			const int32 DX[4] = { 1, -1, 0, 0 };
			const int32 DY[4] = { 0, 0, 1, -1 };

			for (int32 Dir = 0; Dir < 4; ++Dir)
			{
				const int32 NX = X + DX[Dir];
				const int32 NY = Y + DY[Dir];

				if (NX >= 0 && NX < GridSize && NY >= 0 && NY < GridSize)
				{
					const int32 NIdx = NY * GridSize + NX;
					if (!Visited[NIdx] && InputActivations[NIdx] >= RedThreshold)
					{
						Visited[NIdx] = true;
						Queue.Add(NIdx);
					}
				}
			}
		}

		if (ClusterWeight > BestClusterWeight)
		{
			BestClusterWeight = ClusterWeight;
			BestCentroidX = SumX / ClusterWeight;
			BestCentroidY = SumY / ClusterWeight;
		}
	}

	if (BestClusterWeight <= 0.0f)
	{
		UE_LOG(LogTemp, Warning, TEXT("[MLP] Aucun rouge detecte (MaxAct=%.3f, Seuil=%.2f, Pixel=(R:%d G:%d B:%d))."),
			MaxRedActivation, RedThreshold, SamplePixel.R, SamplePixel.G, SamplePixel.B);
		return false;
	}


	OutExpectedX = BestCentroidX;
	OutExpectedY = BestCentroidY;
	return true;
}

void UMLPComponent::RunAIStep()
{
	APawn* OwnerPawn = Cast<APawn>(GetOwner());
	if (!OwnerPawn) return;

	APlayerController* PC = Cast<APlayerController>(OwnerPawn->GetController());
	if (!PC) return;

	// 1. Lire l'image 32x32 et trouver le rouge dans le FOV
	float ExpectedX = 0.5f;
	float ExpectedY = 0.5f;
	if (!ExtractRedInputsAndTarget(ExpectedX, ExpectedY))
	{
		if (bHasTrainingAnchor)
		{
			const float RandomYaw = FMath::FRandRange(-15.0f, 15.0f);
			const float RandomPitch = FMath::FRandRange(-10.0f, 10.0f);
			PC->SetControlRotation(TrainingAnchorRotation + FRotator(RandomPitch, RandomYaw, 0.0f));
		}
		return;
	}

	// 2. Prédiction du réseau de neurones
	ForwardPass();
	const float PredictedX = OutputActivations[0];
	const float PredictedY = OutputActivations[1];

	// 3. Orienter la caméra vers la prédiction
	const float DeltaYaw = (PredictedX - 0.5f) * CameraFOV;
	const float DeltaPitch = (0.5f - PredictedY) * CameraFOV;

	FRotator CurrentRot = PC->GetControlRotation();
	FRotator TargetRot = CurrentRot + FRotator(DeltaPitch, DeltaYaw, 0.0f);
	PC->SetControlRotation(TargetRot);

	// 4. Tir automatique
	FVector CamLoc;
	FRotator CamRot;
	PC->GetPlayerViewPoint(CamLoc, CamRot);

	const FVector AimDir = TargetRot.Vector();
	const FVector TraceEnd = CamLoc + (AimDir * TraceDistance);

	FHitResult HitResult;
	FCollisionQueryParams QueryParams;
	QueryParams.AddIgnoredActor(OwnerPawn);

	const bool bHit = GetWorld()->LineTraceSingleByChannel(HitResult, CamLoc, TraceEnd, ECC_Visibility, QueryParams);
	const bool bHitEnemy = bHit && HitResult.GetActor() && HitResult.GetActor()->ActorHasTag(FName("Enemy"));

	TotalShots++;
	if (bHitEnemy)
	{
		SuccessfulHits++;
	}

	DrawDebugLine(GetWorld(), CamLoc, bHit ? HitResult.ImpactPoint : TraceEnd, bHitEnemy ? FColor::Green : FColor::Red, false, 1.0f, 0, 1.5f);

	// 5. Calcul du coût et Backpropagation
	const float Cost = FMath::Square(ExpectedX - PredictedX) + FMath::Square(ExpectedY - PredictedY);
	if (bTrainOnStep)
	{
		Backpropagate(ExpectedX, ExpectedY);
	}

	const float Accuracy = (static_cast<float>(SuccessfulHits) / static_cast<float>(TotalShots)) * 100.0f;
	UE_LOG(LogTemp, Log, TEXT("[MLP] Tir %d | Resultat: %s | Cost: %.4f | Precision: %.1f%%"),
		TotalShots, bHitEnemy ? TEXT("TOUCHE") : TEXT("RATE"), Cost, Accuracy);

	// 6. Préparer l'angle de la frame suivante pour l'auto-entraînement
	if (bHasTrainingAnchor)
	{
		const float RandomYaw = FMath::FRandRange(-15.0f, 15.0f);
		const float RandomPitch = FMath::FRandRange(-10.0f, 10.0f);
		PC->SetControlRotation(TrainingAnchorRotation + FRotator(RandomPitch, RandomYaw, 0.0f));
	}
}

void UMLPComponent::ToggleAutoTraining()
{
	if (!GetWorld()) return;

	if (GetWorld()->GetTimerManager().IsTimerActive(AutoTrainTimerHandle))
	{
		GetWorld()->GetTimerManager().ClearTimer(AutoTrainTimerHandle);
		bHasTrainingAnchor = false;
		UE_LOG(LogTemp, Log, TEXT("[MLP] Entrainement automatique ARRETE."));
	}
	else
	{
		if (APawn* OwnerPawn = Cast<APawn>(GetOwner()))
		{
			if (APlayerController* PC = Cast<APlayerController>(OwnerPawn->GetController()))
			{
				TrainingAnchorRotation = PC->GetControlRotation();
				bHasTrainingAnchor = true;
			}
		}

		GetWorld()->GetTimerManager().SetTimer(AutoTrainTimerHandle, this, &UMLPComponent::RunAIStep, 0.1f, true);
		UE_LOG(LogTemp, Log, TEXT("[MLP] Entrainement automatique DEMARRE."));
	}
}