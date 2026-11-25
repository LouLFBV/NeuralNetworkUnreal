// Fill out your copyright notice in the Description page of Project Settings.


#include "MLPComponent.h"
#include "Engine/TextureRenderTarget2D.h"
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

void UMLPComponent::RunAIStep()
{
	APawn* OwnerPawn = Cast<APawn>(GetOwner());
	if (!OwnerPawn)
	{
		return;
	}

	APlayerController* PC = Cast<APlayerController>(OwnerPawn->GetController());
	if (!PC)
	{
		return;
	}

	//Repalcer la caméra face aux ennemis avec un léger décalage aléatoire
	if (bHasTrainingAnchor)
	{
		const float RandomYaw = FMath::FRandRange(-15.0f, 15.0f);
		const float RandomPitch = FMath::FRandRange(-10.0f, 10.0f);
		PC->SetControlRotation(TrainingAnchorRotation + FRotator(RandomPitch, RandomYaw, 0.0f));
	}

	// Lire l'image et trouver du rouge
	float ExpectedX = 0.5f;
	float ExpectedY = 0.5f;
	if (!ExtractRedInputsAndTarget(ExpectedX, ExpectedY))
	{
		UE_LOG(LogTemp, Warning, TEXT("[MLP] Aucun ennemi rouge visible dans le champ de vision."));
		return;
	}

	// Prédiction du réseau de neurones 
	ForwardPass();
	const float PredictedX = OutputActivations[0];
	const float PredictedY = OutputActivations[1];

	// Oriente la caméra vers la prédiction
	const float DeltaYaw = (PredictedX - 0.5) * CameraFOV;
	const float DeltaPitch = (0.5 - PredictedY) * CameraFOV;

	FRotator CurrentRotation = PC->GetControlRotation();
	FRotator TargetRotation = CurrentRotation + FRotator(DeltaPitch, DeltaYaw, 0.0f);
	PC->SetControlRotation(TargetRotation);

	// Tir automatique
	FVector CameraLocation;
	FRotator CameraRotation;
	PC->GetPlayerViewPoint(CameraLocation, CameraRotation);

	FVector AimDirection = TargetRotation.Vector();
	FVector TraceEnd = CameraLocation + (AimDirection * TraceDistance);

	FHitResult HitResult;
	FCollisionQueryParams QueryParams;
	QueryParams.AddIgnoredActor(OwnerPawn);

	const bool bHit = GetWorld()->LineTraceSingleByChannel(
		HitResult,
		CameraLocation,
		TraceEnd,
		ECC_Visibility,
		QueryParams
		);

	const bool bHitEnemy = bHit && HitResult.GetActor() && HitResult.GetActor()->ActorHasTag(FName("Enemy"));

	TotalShots++;

	if (bHitEnemy)
	{
		SuccessfulHits++;
	}

	//Debug visuel de tir (vert = touché, jaune = raté)
	DrawDebugLine(GetWorld(), CameraLocation, bHit ? HitResult.ImpactPoint : TraceEnd, bHitEnemy ? FColor::Green : FColor::Yellow, false, 1.0f, 0.0f, 1.5f);

	// Calcul du coût et Backpropagation
	const float Cost = FMath::Square(ExpectedX - PredictedX) + FMath::Square(ExpectedY - PredictedY);
	if (bTrainOnStep)
	{
		BackPropagate(ExpectedX, ExpectedY);
	}

	const float Accuracy = (static_cast<float>(SuccessfulHits)) / (static_cast<float>(TotalShots)) * 100.0f;
	UE_LOG(LogTemp, Log, TEXT("[MLP] Tir : %d | Resultat : %s | Cost : %.4f | Precision : %.1f%%"),
		TotalShots, bHitEnemy ? TEXT("TOUCHE") : TEXT("RATE"), Cost, Accuracy);
}

void UMLPComponent::ToggleAutoTraining()
{
	if (!GetWorld())
	{
		return;
	}

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
		UE_LOG(LogTemp, Log, TEXT("[MLP] Entrainement automatique DEMARRE."))
	}
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

	for (float& W : WeightsInputHidden) 
	{
		W = FMath::FRandRange(-0.02f, 0.02); 
	}
	for (float& B : HiddenBiases) 
	{
		B = 0.0f;
	}
	for (float& W : WeightsHiddenOutput) 
	{
		W = FMath::FRandRange(-0.05f, 0.05f); 
	}
	for (float& B : OutputBiases) 
	{
		B = 0.0f;
	}
}

float UMLPComponent::Sigmoid(float X) const
{
	const float ClampedX = FMath::Clamp(X, -50.0f, 50.0f);
	return 1/(1 + FMath::Exp(-ClampedX));
}

float UMLPComponent::SigmoidDerivate(float SigmoidOutput) const
{
	// Dérivée de la sigmoïde : ResultSigmoid * (1 - ResultSigmoid)
	return SigmoidOutput * (1 - SigmoidOutput);
}

void UMLPComponent::ForwardPass()
{
	// Hidden Layer
	for (int32 h = 0; h < HiddenNeuronCount; ++h)
	{
		float WeightedSum = HiddenBiases[h];
		const float RowOffset = h * InputNeuronCount;

		for (int32 i = 0; i < InputNeuronCount; ++i)
		{
			WeightedSum += InputActivations[i] * WeightsInputHidden[RowOffset + i];
		}

		HiddenActivations[h] = Sigmoid(WeightedSum);
	}

	// Output Layer
	for (int32 o = 0; o < OutputNeuronCount; ++o)
	{
		float WeightedSum = OutputBiases[o];
		const float RowOffset = o * HiddenNeuronCount;

		for (int32 h = 0; h < HiddenNeuronCount; ++h)
		{
			WeightedSum += HiddenActivations[h] * WeightsHiddenOutput[RowOffset + h];
		}

		OutputActivations[o] = Sigmoid(WeightedSum);
	}
}

void UMLPComponent::BackPropagate(float ExpectedX, float ExpectedY)
{
	const float Expected[OutputNeuronCount] = { ExpectedX, ExpectedY };

	// Calcul et Stockage des dérivées de l'Output layer
	for (int32 o = 0; o < OutputNeuronCount; ++o)
	{
		// Dérivée de (Expecteed - Output) ** 2 par rapport à Output pour redescendre de gradient
		const float dCost_dOutput = 2.0f * (OutputActivations[o] - Expected[o]);
		const float dOutput_dWI = SigmoidDerivate(OutputActivations[o]);
		OutputDeltas[o] = dCost_dOutput * dCost_dOutput;
	}

	// Propagation vers le Hidden layer et stockage des valeurs communes
	for (int32 h = 0; h < HiddenNeuronCount; ++h)
	{
		float ErrorSum = 0.0f;
		for (int32 o = 0; o < OutputNeuronCount; ++o)
		{
			ErrorSum += OutputDeltas[o] * WeightsHiddenOutput[o * HiddenNeuronCount + h];
		}
		HiddenDeltas[h] = ErrorSum * SigmoidDerivate(HiddenActivations[h]);
	}

	// Application de la dernière influence et mise à jour des poids/biais

	for (int32 o = 0; o < OutputNeuronCount; ++o)
	{
		const float RowOffset = o * HiddenNeuronCount;
		for (int32 h = 0; h < HiddenNeuronCount; ++h)
		{
			WeightsHiddenOutput[RowOffset + h] -= LearningRate * OutputDeltas[o] * HiddenActivations[h];
		}
		OutputBiases[o] -= LearningRate * OutputDeltas[o];
	}

	for (int32 h = 0; h < HiddenNeuronCount; ++h)
	{
		const float RowOffset = h * InputNeuronCount;
		for (int32 i = 0; i < InputNeuronCount; ++i)
		{
			WeightsInputHidden[RowOffset + i] -= LearningRate * HiddenDeltas[h] * InputActivations[i];
		}
		HiddenBiases[h] -= LearningRate * HiddenDeltas[h];
	}
}

bool UMLPComponent::ExtractRedInputsAndTarget(float& OutputExpectedX, float& OutputExpectedY)
{
	if (!VisionRenderTarget)
	{
		return false;
	}

	FTextureRenderTargetResource* RTRessource = VisionRenderTarget->GameThread_GetRenderTargetResource();
	if (!RTRessource)
	{
		return false;
	}

	TArray<FColor> Pixels;
	if (!RTRessource->ReadPixels(Pixels) || Pixels.Num() < InputNeuronCount)
	{
		return false;
	}

	// Remplir les 1024 neurones d'entrée avec l'intensité de rouge [0, 1]
	for (int32 i = 0; i < InputNeuronCount; ++i)
	{
		const float R = Pixels[i].R / 255.0f;
		const float G = Pixels[i].G / 255.0f;
		const float B = Pixels[i].B / 255.0f;

		//Isoler uniquement la couleur rouge pure
		const float Redness = R - FMath::Max(G, B) * 1.5f;
		InputActivations[i] = FMath::Clamp(Redness, 0.0f, 1.0f);
	}

	// Isoler le plus gros amas de rouge (ennemi le plus proche)
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

			// Les 4 voisins (Haut, Bas, Gauche, Droite)
			const int32 DX[4] = { 1, -1, 0, 0 };
			const int32 DY[4] = { 0, 0, -1, 1 };

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
		return false; // Aucun ennemi rouge dans le champ de vision
	}


	OutputExpectedX = BestCentroidX;
	OutputExpectedY = BestCentroidY;

	return true;
}

