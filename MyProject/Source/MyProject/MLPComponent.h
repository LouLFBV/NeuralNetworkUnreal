// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "MLPComponent.generated.h"

class UTextureRendererTarget2D;

UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class MYPROJECT_API UMLPComponent : public UActorComponent
{
	GENERATED_BODY()

public:	
	UMLPComponent();

protected:
	virtual void BeginPlay() override;

public:	
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MLP|Architecture")
	int32 HiddenNeuronCount = 32;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MLP|Training")
	float LearningRate = 0.15f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MLP|Training")
	bool bTrainOnStep = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MLP|Vision")
	UTextureRendererTarget2D* VisionRenderTarget = nullptr;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MLP|Vision")
	float CameraFOV = 90.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MLP|Vision")
	float RedThreshold = 0.25f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "MLP|Weapon")
	float TraceDistance = 1000.0f;

	UFUNCTION(BlueprintCallable, Category = "MLP")
	void RunAIStep();

	UFUNCTION(BlueprintCallable, Category = "MLP")
	void ToggleAutoTraining();

private :
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

	void InitializeNetwork();
	float Sigmoid(float X) const;
	float SigmoidDerivate(float SigmoidOutput) const;
	void ForwardPass();
	void BackPropagate(float ExpectedX, float ExpectedY);

	bool ExtractRedInputsAndTarget(float& OutputExpectedX, float& OutputExpectedY);
};
