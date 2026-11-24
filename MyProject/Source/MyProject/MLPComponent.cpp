// Fill out your copyright notice in the Description page of Project Settings.


#include "MLPComponent.h"

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
	const float DeltaPitch = (PredictedY - 0.5) * CameraFOV;

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

	if (bHit)
	{
		SuccessfulHits++;
	}

	//Debug visuel de tir (vert = touché, rouge = raté)
	DrawDebugLine(GetWorld(), CameraLocation, bHit ? HitResult.ImpactPoint : TraceEnd, bHitEnemy ? FColor::Green : FColor::Red, false, 1.0f, 0.0f, 1.5f);

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
		UE_LOG(LogTemp, Log, TEXT("[MLP] Entrainement automatique ARRETE."));
	}
	else
	{
		GetWorld()->GetTimerManager().SetTimer(AutoTrainTimerHandle, this, &UMLPComponent::RunAIStep, 0.1f, true);
		UE_LOG(LogTemp, Log, TEXT("[MLP] Entrainement automatique DEMARRE."))
	}
}

void UMLPComponent::InitializeNetwork()
{
	InputActivations.SetNumZeroed(InputNeuronCount);
	HiddenActivations.SetNumZeroed(HiddenNeuronCount);
	OutputActivations.SetNumZeroed(OutputNeuronCount);

	WeightsHiddenOutput.SetNum(InputNeuronCount * HiddenNeuronCount);
	HiddenBiases.SetNum(HiddenNeuronCount);
	WeightsHiddenOutput.SetNum(HiddenNeuronCount * OutputNeuronCount);
	OutputBiases.SetNum(OutputNeuronCount);

	for (float& W : WeightsInputHidden) 
	{
		W = FMath::FRandRange(-0.2f, 0.2); 
	}
	for (float& B : HiddenBiases) 
	{
		B = FMath::FRandRange(-0.1f, 0.1f); 
	}
	for (float& W : WeightsHiddenOutput) 
	{
		W = FMath::FRandRange(-0.2f, 0.2f); 
	}
	for (float& B : OutputBiases) 
	{
		B = FMath::FRandRange(-0.1f, 0.1); 
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
		OutputDeltas[0] = dCost_dOutput * dCost_dOutput;
	}

	// Propagation vers le Hidden layer et stockage des valeurs communes
	for (int32 h = 0; h < HiddenNeuronCount; ++h)
	{
		float ErrorSum = 0.0f;
		for (int32 o = 0; o < OutputNeuronCount; ++o)
		{
			ErrorSum += OutputDeltas[o] + WeightsHiddenOutput[o * HiddenNeuronCount + h];
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
			WeightsInputHidden[RowOffset + i] -= LearningRate * HiddenDeltas[h] * OutputActivations[i];
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

	FTextureRenderTargetResource* RTRessource = VisionRenderTarget->GameThread_GetRenderTargetRessources();
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
	if (int32 i = 0; i < InputNeuronCount; ++i)
	{
		const float R = Pixels[i].R / 255.0f;
		const float G = Pixels[i].G / 255.0f;
		const float B = Pixels[i].B / 255.0f;

		//Isoler uniquement la couleur rouge pure
		InputActivations[i] = FMath::Max(0.0f, R - (G + B) * 0.5f);
	}

	return false;
}

