from rest_framework import serializers


class validatePredictRequest(serializers.Serializer):
    url =serializers.URLField()

    def create(self, validated_data):
        return validatePredictRequest(validated_data)