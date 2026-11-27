// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "MLPComponent.generated.h"

class UTextureRenderTarget2D;
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnHit);

UCLASS(ClassGroup = (Custom), meta = (BlueprintSpawnableComponent))
class MYPROJECT_API UMLPComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UMLPComponent();

protected:
	virtual void BeginPlay() override;

public:
	UPROPERTY(BlueprintAssignable, Category = "MLP|Events")
	FOnHit OnHit;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MLP|Architecture")
	int32 HiddenNeuronCount = 32;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MLP|Training")
	float LearningRate = 0.5f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MLP|Training")
	bool bTrainOnStep = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MLP|Vision")
	UTextureRenderTarget2D* VisionRenderTarget = nullptr;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MLP|Vision")
	float CameraFOV = 90.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MLP|Vision")
	float RedThreshold = 0.15f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MLP|Weapon")
	float TraceDistance = 1000.0f;

	UFUNCTION()
	void SetHasWeapon(bool hasWeapon) { bHasWeapon = hasWeapon; };

	UFUNCTION(BlueprintCallable, Category = "MLP")
	void RunAIStep();

	UFUNCTION(BlueprintCallable, Category = "MLP")
	void ToggleAutoTraining();

private:
	static constexpr int32 GridSize = 32;
	static constexpr int32 InputNeuronCount = GridSize * GridSize;
	static constexpr int32 OutputNeuronCount = 2;

	TArray<float> InputActivations;
	TArray<float> HiddenActivations;
	TArray<float> OutputActivations;

	TArray<float> WeightsInputHidden;
	TArray<float> HiddenBiases;
	TArray<float> WeightsHiddenOutput;
	TArray<float> OutputBiases;

	TArray<float> HiddenDeltas;
	TArray<float> OutputDeltas;

	int32 TotalShots = 0;
	int32 SuccessfulHits = 0;
	FTimerHandle AutoTrainTimerHandle;

	bool bHasTrainingAnchor = false;
	FRotator TrainingAnchorRotation = FRotator::ZeroRotator;

	void InitializeNetwork();
	float Sigmoid(float X) const;
	float SigmoidDerivative(float SigmoidOutput) const;
	void ForwardPass();
	void Backpropagate(float ExpectedX, float ExpectedY);

	bool ExtractRedInputsAndTarget(float& OutputExpectedX, float& OutputExpectedY);

	bool bHasWeapon = false;
};