pipeline{
  agent any{
    trigger{
      githubPush()
    }
    stages{
      stage('Checkout'){
        steps{
          checkout scm
        }
      }
    }
  }
  
